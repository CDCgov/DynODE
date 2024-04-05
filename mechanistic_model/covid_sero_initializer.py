import os

import numpy as np
import pandas as pd

import utils
from config.config import Config
from mechanistic_model.covid_initializer import CovidInitializer


class CovidSeroInitializer(CovidInitializer):
    def __init__(self, config_initializer_path, global_variables_path):
        """
        initialize a mechanistic model for covid19 case prediction using serological data.
        """
        initializer_json = open(config_initializer_path, "r").read()
        global_json = open(global_variables_path, "r").read()
        self.config = Config(global_json).add_file(initializer_json)

        if not hasattr(self.config, "INITIAL_POPULATION_FRACTIONS"):
            self.load_initial_population_fractions()

        self.config.POPULATION = (
            self.config.POP_SIZE * self.config.INITIAL_POPULATION_FRACTIONS
        )
        # self.POPULATION.shape = (NUM_AGE_GROUPS,)

        if not hasattr(self.config, "INIT_IMMUNE_HISTORY"):
            self.load_immune_history_via_serological_data()
        # self.INIT_IMMUNE_HISTORY.shape = (age, hist, num_vax, waning)
        if not hasattr(self.config, "CONTACT_MATRIX"):
            self.load_contact_matrix()

        if not hasattr(self.config, "CROSSIMMUNITY_MATRIX"):
            self.load_cross_immunity_matrix()
        # stratify initial infections appropriately across age, hist, vax counts
        if not (
            hasattr(self.config, "INIT_INFECTED_DIST")
            and hasattr(self.config, "INIT_EXPOSED_DIST")
        ) and hasattr(self.config, "CONTACT_MATRIX_PATH"):
            self.load_init_infection_infected_and_exposed_dist_via_contact_matrix()

        # load initial state using INIT_IMMUNE_HISTORY, INIT_INFECTED_DIST, and INIT_EXPOSED_DIST
        self.INITIAL_STATE = self.load_initial_state(
            self.config.INITIAL_INFECTIONS
        )

    def load_immune_history_via_serological_data(self):
        """
        loads the sero init file for self.config.REGIONS[0] and converts it to a numpy matrix
        representing the initial immune history of the individuals in the system. Saving matrix
        to self.config.INIT_IMMUNE_HISTORY

        assumes each age bin in INIT_IMMUNE_HISTORY sums to 1, will fail if not.
        """
        file_name = str(self.config.REGIONS[0]).replace(" ", "_") + "_sero.csv"
        sero_path = os.path.join(self.config.SEROLOGICAL_DATA_PATH, file_name)
        assert os.path.exists(sero_path), (
            "sero path for %s does not exist" % sero_path
        )
        # read in csv, do a bunch of data cleaning
        sero_df = pd.read_csv(sero_path)
        sero_df["type"] = sero_df["type"].fillna("None")
        sero_df.columns = ["age", "hist", "vax", "0", "1", "2", "3", "4"]
        melted_df = pd.melt(
            sero_df,
            id_vars=["age", "hist", "vax"],
            value_vars=["0", "1", "2", "3", "4"],
            var_name="wane",
            value_name="val",
        )
        melted_df["wane"] = melted_df["wane"].astype("int64")
        converter = {"None": 0, "PreOmicron": 1, "Omicron": 2, "Both": 3}
        melted_df["hist"] = melted_df["hist"].apply(converter.get)
        melted_df["age"] = melted_df["age"].apply(
            self.config.AGE_GROUP_IDX._member_map_.get
        )
        # we now have a pandas df representing the values of each of our 4 dimensions
        matrix_shape = (
            len(melted_df["age"].unique()),
            len(melted_df["hist"].unique()),
            len(melted_df["vax"].unique()),
            len(melted_df["wane"].unique()),
        )

        indices = np.indices((matrix_shape)).reshape(4, -1).T
        # our matrix may be larger than the sero data we have access to
        # if we tracking more strains than the initialization data provides etc.
        sero_matrix = np.zeros(
            (
                self.config.NUM_AGE_GROUPS,
                2**self.config.NUM_STRAINS,
                self.config.MAX_VAX_COUNT + 1,
                self.config.NUM_WANING_COMPARTMENTS,
            )
        )
        # go through each dimension specified, fill in the value into numpy matrix at `val`
        for i in range(indices.shape[0]):
            age, hist, vax, wane = indices[i]
            val = melted_df.loc[
                (melted_df["age"] == age)
                & (melted_df["hist"] == hist)
                & (melted_df["vax"] == vax)
                & (melted_df["wane"] == wane),
                "val",
            ].iloc[0]
            sero_matrix[age, hist, vax, wane] = val

        # ages must all sum to 1
        assert np.isclose(
            np.sum(
                sero_matrix,
                axis=(
                    self.config.S_AXIS_IDX.hist,
                    self.config.S_AXIS_IDX.vax,
                    self.config.S_AXIS_IDX.wane,
                ),
            ),
            [1.0] * self.config.NUM_AGE_GROUPS,
        ).all(), (
            "each age group does not sum to 1 in the sero initialization file of %s"
            % str(self.config.REGIONS[0])
        )
        self.config.INIT_IMMUNE_HISTORY = sero_matrix

    def load_init_infection_infected_and_exposed_dist_via_contact_matrix(self):
        """
        a function which estimates the demographics of initial infections by looking at the currently susceptible population's
        level of protection as well as the contact matrix for mixing patterns.

        Disperses these infections across the E and I compartments by the ratio of the waiting times in each compartment.
        """
        # use relative wait times in each compartment to get distribution of infections across
        # infected vs exposed compartments
        exposed_to_infectous_ratio = (
            self.config.EXPOSED_TO_INFECTIOUS / self.config.INFECTIOUS_PERIOD
        )
        # use contact matrix to get the infection age distributions
        eig_data = np.linalg.eig(self.config.CONTACT_MATRIX)
        max_index = np.argmax(eig_data[0])
        infection_age_dists = abs(eig_data[1][:, max_index])
        # normalize so they sum to 1
        infection_age_dists = infection_age_dists / np.sum(infection_age_dists)
        # we will now weight the population in each strata of the S compartment by their
        # relative susceptibility to the incoming omicron strain
        crossimmunity_matrix = self.config.CROSSIMMUNITY_MATRIX[
            self.config.STRAIN_IDX.omicron, :
        ]
        # p.vax_susceptibility_strain.shape = (MAX_VAX_COUNT,)
        vax_efficacy_strain = self.config.VAX_EFF_MATRIX[
            self.config.STRAIN_IDX.omicron, :
        ]
        # susceptibility by hist and vax status
        initial_immunity = 1 - np.einsum(
            "j, k",
            1 - crossimmunity_matrix,
            1 - vax_efficacy_strain,
        )
        # susceptibility by hist, vax, and wane status
        waned_immunity = np.einsum(
            "jk,l", initial_immunity, self.config.WANING_PROTECTIONS
        )  # shape=(hist, vax, wane)
        # for each age bin, multiply the pop counts of each
        # hist, vax, wane strata by the protect afforded by that strata
        pop_sizes_with_relative_immunity = waned_immunity[np.newaxis, ...] * (
            self.config.POPULATION[:, np.newaxis, np.newaxis, np.newaxis]
            * self.config.INIT_IMMUNE_HISTORY
        )  # shape(age, hist, vax, wane)

        # normalize so each age bin in pop_sizes_with_relative_immunity sums to 1
        pop_sizes_with_relative_immunity = (
            pop_sizes_with_relative_immunity
            / np.sum(
                pop_sizes_with_relative_immunity, axis=(1, 2, 3), keepdims=True
            )
        )
        # bring back the infection age distributions from the contact matrix eig value
        # since sum(infection_age_dists) = 1, it is now the case that sum(pop_sizes_with_relative_immunity) = 1
        pop_sizes_with_relative_immunity = (
            infection_age_dists[:, np.newaxis, np.newaxis, np.newaxis]
            * pop_sizes_with_relative_immunity
        )
        # sum across wane axis since we dont track that for E or I compartments
        infection_dist = np.sum(pop_sizes_with_relative_immunity, axis=-1)
        infection_dist = np.repeat(
            infection_dist[:, :, :, np.newaxis], repeats=3, axis=-1
        )
        # set all strains that arent omicron to zero, since we repeated them 3 times above
        infection_dist[
            :,
            :,
            :,
            list(
                set(list(range(self.config.NUM_STRAINS)))
                - set([self.config.STRAIN_IDX.omicron])
            ),
        ] = 0
        # disperse infections across E and I compartments by exposed_to_infectous_ratio
        self.config.INIT_INFECTION_DIST = infection_dist
        self.config.INIT_EXPOSED_DIST = (
            exposed_to_infectous_ratio * self.config.INIT_INFECTION_DIST
        )
        self.config.INIT_INFECTED_DIST = (
            1 - exposed_to_infectous_ratio
        ) * self.config.INIT_INFECTION_DIST

    def load_contact_matrix(self):
        """
        a wrapper function that loads a contact matrix for the USA based on mixing paterns data found here:
        https://github.com/mobs-lab/mixing-patterns

        Updates
        ----------
        `self.config.CONTACT_MATRIX` : numpy.ndarray
            a matrix of shape (self.config.NUM_AGE_GROUPS, self.config.NUM_AGE_GROUPS) with each value representing TODO
        """
        self.config.CONTACT_MATRIX = utils.load_demographic_data(
            self.config.DEMOGRAPHIC_DATA_PATH,
            self.config.REGIONS,
            self.config.NUM_AGE_GROUPS,
            self.config.AGE_LIMITS[0],
            self.config.AGE_LIMITS,
        )[self.config.REGIONS[0]]["avg_CM"]

    def load_cross_immunity_matrix(self):
        """
        Loads the Crossimmunity matrix given the strain interactions matrix.
        Strain interactions matrix is a matrix of shape (num_strains, num_strains) representing the relative immune escape risk
        of those who are being challenged by a strain in dim 0 but have recovered from a strain in dim 1.
        Neither the strain interactions matrix nor the crossimmunity matrix take into account waning.

        Updates
        ----------
        self.config.CROSSIMMUNITY_MATRIX:
            updates this matrix to shape (self.config.NUM_STRAINS, self.config.NUM_PREV_INF_HIST) containing the relative immune escape
            values for each challenging strain compared to each prior immune history in the model.
        """
        self.config.CROSSIMMUNITY_MATRIX = (
            utils.strain_interaction_to_cross_immunity(
                self.config.NUM_STRAINS, self.config.STRAIN_INTERACTIONS
            )
        )
