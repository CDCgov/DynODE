"""
A script designed to bypass the usual process of creating an experiment, launching it to azure, and collecting projections

Designed to run a single projection config across 50 states, based on the ending point
of HISTORICAL_FIT_PATH

"""

import json
import os

import pandas as pd

from config.config import Config
from exp.projections_2407_2507.inferer_projection import ProjectionParameters
from experiment_setup import code_to_state, get_all_codes
from mechanistic_model.abstract_azure_runner import AbstractAzureRunner
from mechanistic_model.mechanistic_runner import MechanisticRunner
from model_odes.seip_model import seip_ode2

HISTORICAL_FIT_PATH = (
    "/output/fifty_state_6strain_2202_2407/SMH_6strains_240731"
)
PROJECTION_PATH = "exp/projections_2407_2507/local/local_projection.json"
GLOBAL_PATH = "exp/projections_2407_2507/local/config_global.json"
state_names = pd.read_csv("data/fips_to_name.csv")
OUTPUT_PATH = "/output/local_projections/test_1"


class LocalProjector(AbstractAzureRunner):
    def __init__(self, azure_output_dir, historical_fit_path):
        # its still called azure_output_dir but its still local
        self.azure_output_dir = azure_output_dir
        self.historical_fit_path = historical_fit_path
        if not os.path.exists(azure_output_dir):
            os.makedirs(azure_output_dir, exist_ok=True)
        # create two dual loggers to save sys.stderr and sys.stdout to files in `azure_output_dir`

    def process_state(
        self, state, projection_config_path: str, global_config_path: str
    ):
        projection_period_num_days = 365
        posteriors_path = os.path.join(
            self.historical_fit_path,
            state,
        )
        checkpoint_path = os.path.join(posteriors_path, "checkpoint.json")
        assert os.path.exists(checkpoint_path), (
            "checkpoint does not exist for this state %s" % state
        )
        posteriors = json.load(open(checkpoint_path, "r"))
        # the final states of the fitting period are saved within posteriors
        print("Running the following state: " + state + "\n")
        # modify the global/projection configs to get the right state in there

        runner = MechanisticRunner(seip_ode2)
        # small override needed to change the REGIONS tag
        inferer = LocalProjectionParameters(
            global_config_path, projection_config_path, runner, state=state
        )
        timeseries_save_path = os.path.join(self.azure_output_dir, state)
        if not os.path.exists(timeseries_save_path):
            os.makedirs(timeseries_save_path)
        self.save_inference_timelines(
            inferer,
            particles_saved=1,
            timeline_filename="%s/azure_visualizer_timeline.csv" % state,
            tf=projection_period_num_days,
            external_particle=posteriors,
        )


class LocalProjectionParameters(ProjectionParameters):
    def __init__(
        self,
        global_variables_path: str,
        distributions_path: str,
        runner: MechanisticRunner,
        state: str,
        prior_inferer=None,
    ):
        """A specialized init method which does not take an initial state, this is because
        posterior particles will contain the initial state used."""
        distributions_json = open(distributions_path, "r").read()
        global_json = open(global_variables_path, "r").read()
        # modify our JSONS since we getting rid of experiment creation in this local run
        # with a proper regions tag we should be able to load the correct contact matricies
        self.config = Config(global_json).add_file(distributions_json)
        self.config.__dict__["REGIONS"] = [code_to_state(state, state_names)]
        self.runner = runner
        self.infer_complete = False  # flag once inference completes
        self.set_infer_algo(prior_inferer=prior_inferer)
        self.load_vaccination_model()
        self.load_contact_matrix()


def collate_timeseries(output_path):
    states = os.listdir(output_path)
    print(states)
    dfs = list()
    for st in states:
        state_path = os.path.join(output_path, st)
        csv_path = os.path.join(state_path, "azure_visualizer_timeline.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(
                csv_path,
                usecols=[
                    "chain_particle",
                    "date",
                    "pred_hosp_0_17",
                    "pred_hosp_18_49",
                    "pred_hosp_50_64",
                    "pred_hosp_65+",
                    "JN1_strain_proportion",
                    "KP_strain_proportion",
                    "X_strain_proportion",
                ],
            )

            df["state"] = st
            dfs.append(df)

    final_df = pd.concat(dfs)
    final_df.to_csv(
        os.path.join(output_path, "projections_local.csv"), index=False
    )
    return final_df


states = get_all_codes(state_names)
states.remove("US")
states.remove("DC")
# states = ["AL", "CA"]
local_projector = LocalProjector(OUTPUT_PATH, HISTORICAL_FIT_PATH)
for state in states:
    # use the same projection/global config for each state
    # the projector will slot in the state name at runtime
    # to load the correct contact matricies etc
    # state_solution_dct[state] = projector.process_state(
    #     state, PROJECTION_PATH, GLOBAL_PATH
    # )
    local_projector.process_state(state, PROJECTION_PATH, GLOBAL_PATH)

collate_timeseries(local_projector.azure_output_dir)
