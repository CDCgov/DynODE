import json
import os

import numpy as np
import pandas as pd

import mechanistic_model.utils as utils
from config.config import Config
from mechanistic_model.covid_initializer import CovidInitializer


class smh_initializer_epoch_two(CovidInitializer):
    def __init__(self, config_initializer_path, global_variables_path):
        initializer_json = open(config_initializer_path, "r").read()
        global_json = open(global_variables_path, "r").read()
        self.config = Config(global_json).add_file(initializer_json)
        assert hasattr(
            self.config, "PREV_EPOCH_CHECKPOINT_PATH"
        ), "This special class of initializer requires a previous epoch to initalize from"
        self.load_average_april_state()

    def load_average_april_state(self):
        region_to_usps = pd.read_csv(self.config.STATE_NAMES)
        region_name = self.config.REGIONS[0]
        usps_code = region_to_usps[region_to_usps["stname"] == region_name][
            "stusps"
        ].values[0]
        # find the checkpoint file from the previous epoch in the output container
        state_checkpoint_file = os.path.join(
            self.config.PREV_EPOCH_CHECKPOINT_PATH,
            "%s/checkpoint.json" % usps_code,
        )
        posterior_samples = json.load(open(state_checkpoint_file, "r"))
        init_state = []
        for compartment_name in self.config.COMPARTMENT_IDX._member_names_:
            saved_posterior_name = (
                "april_1st_timestep_%s" % compartment_name.lower()
            )
            averaged_compartment = np.average(
                np.array(posterior_samples[saved_posterior_name]), axis=(0, 1)
            )
            # now we combine all 3 strains in the finals state into 1
            state_mapping, strain_mapping = utils.combined_strains_mapping(
                1, 0, 3
            )
            # if S compartment, the last axis is not `strain`, which is important for combining
            strain_axis = compartment_name != "S"
            # run the first time to combine strain 1 and 0
            average_compartment_combined = utils.combine_strains(
                averaged_compartment,
                num_strains=self.config.NUM_STRAINS,
                state_mapping=state_mapping,
                strain_mapping=strain_mapping,
                strain_axis=strain_axis,
            )
            # run this twice to combine all strains into the 0th strain
            # keep the same state and strain mapping since we shifted strain 2 -> 1 above
            average_compartment_combined = utils.combine_strains(
                average_compartment_combined,
                num_strains=self.config.NUM_STRAINS,
                state_mapping=state_mapping,
                strain_mapping=strain_mapping,
                strain_axis=strain_axis,
            )
            if compartment_name == "C":
                average_compartment_combined = np.zeros(
                    average_compartment_combined.shape
                )
            init_state.append(average_compartment_combined)
        self.INITIAL_STATE = tuple(init_state)
