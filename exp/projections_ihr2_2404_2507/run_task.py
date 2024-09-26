# ruff: noqa: E402
import argparse
import json
import os
import shutil
import sys

import jax
import numpy as np

# adding things to path since in a docker container pathing gets changed
sys.path.append("/app/")
sys.path.append("/input/exp/fifty_state_5strain_2202_2404/")
print(os.getcwd())
from resp_ode import MechanisticRunner
from resp_ode.model_odes.seip_model_flatten_immune_hist import seip_ode
from src.mechanistic_azure.abstract_azure_runner import AbstractAzureRunner

# sys.path.append(".")
# sys.path.append(os.getcwd())
from .inferer_projection import ProjectionParameters

jax.config.update("jax_enable_x64", True)

# will be multiplied by number of chains to get total number of posteriors
NUM_SAMPLES_PER_STATE_PER_SCENARIO = 25
HISTORICAL_FIT_PATH = (
    "/output/fifty_state_5strain_2202_2404/SMH_5strains_240807_v16"
)
EXP_ID = "projections_ihr2_2404_2507"


class ProjectionRunner(AbstractAzureRunner):
    # __init__ already implemented by the abstract case
    def __init__(self, azure_output_dir):
        super().__init__(azure_output_dir)

    def process_state(self, state, jobid=None, local_run=False, scenario=None):
        projection_period_num_days = 434
        posteriors_path = os.path.join(
            HISTORICAL_FIT_PATH,
            state,
        )
        checkpoint_path = os.path.join(posteriors_path, "checkpoint.json")
        assert os.path.exists(checkpoint_path), (
            "checkpoint does not exist for this state %s" % state
        )
        posteriors = json.load(open(checkpoint_path, "r"))
        # the final states of the fitting period are saved within posteriors
        # step 1: define your paths, now in the input
        state_config_path = os.path.join(
            f"/input/exp/{EXP_ID}/{jobid}/states",
            state,
        )
        if local_run:
            state_config_path = os.path.join(
                f"/input/exp/{EXP_ID}/states",
                state,
            )
        assert os.path.exists(state_config_path), (
            "the state path %s does not exist" % state_config_path
        )
        print("Running the following state: " + state + "\n")
        # global_config include definitions such as age bin bounds and strain definitions
        # Any value or data structure that needs context to be interpretted is here.
        GLOBAL_CONFIG_PATH = os.path.join(
            state_config_path, "config_global.json"
        )
        # a config file that defines the scenario being run
        INFERER_CONFIG_PATH = os.path.join(
            state_config_path, "%s.json" % scenario
        )

        cg_path = os.path.join(
            self.azure_output_dir, "config_global_used.json"
        )
        ci_path = os.path.join(
            self.azure_output_dir, "config_inferer_used.json"
        )
        # if you are hitting this block, this means you are either running locally
        # or you rerunning a jobid which is bad practice
        if os.path.exists(cg_path):
            print(
                "You are overriding an existing job's outputs, "
                "this is bad practice and can destroy reproducibility. Proceed with caution"
            )
            os.remove(cg_path)
        if os.path.exists(ci_path):
            os.remove(ci_path)
        shutil.copy(GLOBAL_CONFIG_PATH, cg_path)
        shutil.copy(INFERER_CONFIG_PATH, ci_path)

        # sets up the initial conditions, initializer.get_initial_state() passed to runner
        runner = MechanisticRunner(seip_ode)
        inferer = ProjectionParameters(
            GLOBAL_CONFIG_PATH, INFERER_CONFIG_PATH, runner
        )
        # self.save_inference_posteriors(inferer)
        np.random.seed(4326)
        self.save_inference_timelines(
            inferer,
            particles_saved=NUM_SAMPLES_PER_STATE_PER_SCENARIO,
            external_particle=posteriors,
            tf=projection_period_num_days,
        )


parser = argparse.ArgumentParser()
parser.add_argument(
    "-s",
    "--state",
    type=str,
    help="directory for the state to run, resembles USPS code of the state",
)

parser.add_argument(
    "-j", "--jobid", type=str, help="job-id of the state being run on Azure"
)
parser.add_argument(
    "-sc", "--scenario", type=str, help="scenario being run on Azure"
)
parser.add_argument(
    "-l", "--local", action="store_true", help="scenario being run on Azure"
)

if __name__ == "__main__":
    args = parser.parse_args()
    jobid: str = args.jobid
    state: str = args.state
    scenario: str = args.scenario
    local: bool = args.local
    # we are going to be rerouting stdout and stderror to files in our output blob
    # stdout = sys.stdout
    # stderror = sys.stderr
    save_path = "/output/%s/%s/%s/%s/" % (
        EXP_ID,
        jobid,
        state,
        scenario,
    )
    runner = ProjectionRunner(save_path)
    runner.process_state(state, jobid, local_run=local, scenario=scenario)
