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
sys.path.append("/input/exp/fifty_state_6strain_2202_2407/")
print(os.getcwd())
# sys.path.append(".")
# sys.path.append(os.getcwd())
import jax.numpy as jnp
import pandas as pd
from inferer_projection import ProjectionParameters

from mechanistic_model.abstract_azure_runner import AbstractAzureRunner
from mechanistic_model.covid_sero_initializer import CovidSeroInitializer
from mechanistic_model.mechanistic_runner import MechanisticRunner
from mechanistic_model.utils import combine_strains, combined_strains_mapping
from model_odes.seip_model import seip_ode2

jax.config.update("jax_enable_x64", True)
# SCENARIOS:
# scenario_vax1_it32_ve40
# scenario_vax1_it93_ve40

# will be multiplied by number of chains to get total number of posteriors
NUM_SAMPLES_PER_STATE_PER_SCENARIO = 25


class ProjectionRunner(AbstractAzureRunner):
    # __init__ already implemented by the abstract case
    def __init__(self, azure_output_dir):
        super().__init__(azure_output_dir)

    def process_state(
        self, state, jobid=None, jobid_in_path=False, scenario=None
    ):
        fitting_period_num_days = 870
        projection_period_num_days = 365
        posteriors_path = os.path.join(
            "/output/fifty_state_6strain_2202_2407/scen_6str_v1_2",
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
            "/input/exp/projections_2407_2507/%s/states" % jobid,
            state,
        )
        # state_config_path = "exp/fifty_state_sero_second_try/" + args.state + "/"
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
        # save copies of the used config files to output for reproducibility purposes
        shutil.copy(
            GLOBAL_CONFIG_PATH,
            self.azure_output_dir + "config_global_used.json",
        )
        shutil.copy(
            INFERER_CONFIG_PATH,
            self.azure_output_dir + "config_inferer_used.json",
        )
        # sets up the initial conditions, initializer.get_initial_state() passed to runner
        runner = MechanisticRunner(seip_ode2)
        inferer = ProjectionParameters(
            GLOBAL_CONFIG_PATH, INFERER_CONFIG_PATH, runner
        )
        # self.save_inference_posteriors(inferer)
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

if __name__ == "__main__":
    args = parser.parse_args()
    jobid = args.jobid
    state = args.state
    scenario = args.scenario
    # we are going to be rerouting stdout and stderror to files in our output blob
    # stdout = sys.stdout
    # stderror = sys.stderr
    save_path = "/output/projections_2407_2507/%s/%s/%s/" % (
        jobid,
        state,
        scenario,
    )
    runner = ProjectionRunner(save_path)
    runner.process_state(state, jobid, jobid_in_path=False, scenario=scenario)
