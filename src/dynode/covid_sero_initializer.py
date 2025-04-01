"""Define a covid initializer for parsing and transforming input serology data."""

import os

import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import Array
from . import utils, logger
from .abstract_initializer import AbstractInitializer
from .config import Config
from .typing import SEIC_Compartments


class CovidSeroInitializer(AbstractInitializer):
    """A Covid specific initializer class using serology input data to stratify immunity."""

    def __init__(self, config_initializer_path, global_variables_path):
        """Create an initializer for covid19 case prediction using serological data.

        Updates the `self.INITIAL_STATE` jax array to contain all relevant
        age and immune distributions of the specified population.

        Parameters
        ----------
        config_initializer_path : str
            Path to initializer specific JSON parameters.
        global_variables_path : str
            Path to global JSON for parameters shared across all components
            of the model.
        """
        initializer_json = open(config_initializer_path, "r").read()
        global_json = open(global_variables_path, "r").read()
        self.config = Config(global_json).add_file(initializer_json)

        if not hasattr(self.config, "INITIAL_POPULATION_FRACTIONS"):
            self.config.INITIAL_POPULATION_FRACTIONS = (
                self.load_initial_population_fractions()
            )

        self.config.POPULATION = (
            self.config.POP_SIZE * self.config.INITIAL_POPULATION_FRACTIONS
        )
        # self.POPULATION.shape = (NUM_AGE_GROUPS,)

        if not hasattr(self.config, "INIT_IMMUNE_HISTORY"):
            self.config.INIT_IMMUNE_HISTORY = (
                self.load_immune_history_via_serological_data()
            )
        # self.INIT_IMMUNE_HISTORY.shape = (age, hist, num_vax, waning)
        if not hasattr(self.config, "CONTACT_MATRIX"):
            self.config.CONTACT_MATRIX = self.load_contact_matrix()

        if not hasattr(self.config, "CROSSIMMUNITY_MATRIX"):
            self.config.CROSSIMMUNITY_MATRIX = (
                self.load_cross_immunity_matrix()
            )
        # stratify initial infections appropriately across age, hist, vax counts
        if not (
            hasattr(self.config, "INIT_INFECTIOUS_DIST")
            and hasattr(self.config, "INIT_EXPOSED_DIST")
        ) and hasattr(self.config, "CONTACT_MATRIX_PATH"):
            self.config.INIT_INFECTION_DIST = (
                self.load_initial_infection_dist_via_contact_matrix()
            )
            self.config.INIT_INFECTIOUS_DIST = (
                self.get_initial_infectious_distribution()
            )
            self.config.INIT_EXPOSED_DIST = (
                self.get_initial_exposed_distribution()
            )

        # load initial state using
        # INIT_IMMUNE_HISTORY, INIT_INFECTIOUS_DIST, and INIT_EXPOSED_DIST
        self.INITIAL_STATE = self.load_initial_state(
            self.config.INITIAL_INFECTIONS
        )

    def load_initial_state(
        self, initial_infections: float
    ) -> SEIC_Compartments:
        """Disperse initial infections across infectious and exposed compartments.

        Parameters
        ----------
        initial_infections: the number of infections to
        disperse between infectious and exposed compartments.

        Returns
        -------
        INITIAL_STATE: SEIC_Compartments
            a tuple of len 4 representing the S, E, I, and C compartment
            population counts after model initialization.

        Notes
        -----
        Requires the following variables be loaded into self:
        - CONTACT_MATRIX: loading in config or via
        `self.load_contact_matrix()`
        - INIT_INFECTIOUS_DIST: loaded in config or via
        `get_initial_infectious_distribution()`
        - INIT_EXPOSED_DIST: loaded in config or via
        `get_initial_exposed_distribution()`
        - INIT_IMMUNE_HISTORY: loaded in config or via
        `load_immune_history_via_serological_data()`.

        Age and immune history distributions of infectious and exposed
        populations dictated by `self.config.INIT_INFECTIOUS_DIST` and
        `self.config.INIT_EXPOSED_DIST` matricies. Subtracts both those
        populations from the total population and places the remaining
        individuals in the susceptible compartment, distributed according to
        the `self.config.INIT_IMMUNE_HISTORY` matrix.
        """
        # create population distribution with INIT_INFECTIOUS_DIST then sum by age
        initial_infectious_count = (
            initial_infections * self.config.INIT_INFECTIOUS_DIST
        )
        initial_infectious_count_ages = jnp.sum(
            initial_infectious_count,
            axis=(
                self.config.I_AXIS_IDX.hist,
                self.config.I_AXIS_IDX.vax,
                self.config.I_AXIS_IDX.strain,
            ),
        )
        # create population distribution with INIT_EXPOSED_DIST then sum by age
        initial_exposed_count = (
            initial_infections * self.config.INIT_EXPOSED_DIST
        )
        initial_exposed_count_ages = jnp.sum(
            initial_exposed_count,
            axis=(
                self.config.I_AXIS_IDX.hist,
                self.config.I_AXIS_IDX.vax,
                self.config.I_AXIS_IDX.strain,
            ),
        )
        # susceptible / partial susceptible =
        # Total population - infected_count - exposed_count
        initial_susceptible_count = (
            self.config.POPULATION
            - initial_infectious_count_ages
            - initial_exposed_count_ages
        )[
            :, jnp.newaxis, jnp.newaxis, jnp.newaxis
        ] * self.config.INIT_IMMUNE_HISTORY
        # cumulative count always starts at zero
        return (
            initial_susceptible_count,  # s
            initial_exposed_count,  # e
            initial_infectious_count,  # i
            jnp.zeros(initial_exposed_count.shape),  # c
        )

    def load_immune_history_via_serological_data(self) -> np.ndarray:
        """Load the serology init file and calculates initial immune history of susceptibles.

        Returns
        -------
        np.ndarray
            The initial immune history of individuals within each age bin in the system.
            `INIT_IMMUNE_HISTORY[i][j][k][l]` describes the proportion of
            individuals in age bin `i`, who fall under
            immune history `j`, vaccination count `k`, and waning bin `l`.

        Examples
        --------
        Assume united_states_initialization.csv exists and is valid.
        >>> init = CovidSeroInitializer("c.json", "global.json")
        >>> immune_histories = init.load_immune_history_via_serological_data()
        >>> immune_histories.shape == (init.config.NUM_AGE_GROUPS,
        ...                            2**init.config.NUM_STRAINS,
        ...                            init.config.MAX_VACCINATION_COUNT + 1,
        ...                            init.config.NUM_WANING_COMPARTMENTS)
        True
        # sum across all bins except age group and ensure they sum to 1
        >>> all(np.isclose(np.sum(immune_histories, axis=(1, 2, 3)),
        ...                       np.ones(init.config.NUM_AGE_GROUPS)))
        True
        """
        file_name = (
            str(self.config.REGIONS[0]).replace(" ", "_")
            + "_initialization.csv"
        )
        sero_path = os.path.join(self.config.SEROLOGICAL_DATA_PATH, file_name)
        assert os.path.exists(sero_path), (
            "sero path for %s does not exist" % sero_path
        )
        # read in csv, do a bunch of data cleaning
        sero_df = pd.read_csv(sero_path)
        sero_df["type"] = sero_df["type"].fillna("None")
        sero_df.columns = pd.Index(
            ["age", "hist", "vax", "0", "1", "2", "3", "4"]
        )
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
                self.config.MAX_VACCINATION_COUNT + 1,
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
        return sero_matrix

    def load_initial_infection_dist_via_contact_matrix(
        self,
    ) -> np.ndarray:
        """Estimates the demographics and immune histories of initial infections.

        Looks at the currently susceptible population's proposed level of
        protection as well as the contact matrix for mixing patterns. Tailored
        specifically for initialization with the `omicron` strain for
        feburary 2022.

        Returns
        -------
        np.ndarray
            matrix describing the proportion of new infections falling under
            each stratification of the compartment. E.g
            `INIT_INFECTION_DIST[i][j][k][l]` describes the proportion of
            individuals in age bin `i`, who fall under
            immune history `j`, vaccination count `k`, and strain`l`
        """
        if not hasattr(self.config, "CONTACT_MATRIX_PATH"):
            raise RuntimeError(
                "Attempting to build initial infection distribution "
                "without a path to a contact matrix in "
                "self.config.CONTACT_MATRIX_PATH"
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
        vax_efficacy_strain = self.config.VACCINE_EFF_MATRIX[
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
        pop_sizes_with_relative_immunity = (
            1 - waned_immunity[np.newaxis, ...]
        ) * (
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
        return infection_dist

    def get_initial_infectious_distribution(self) -> np.ndarray:
        """Get actively infectious proportion of initial infections.

        Returns
        -------
        np.ndarray
            actively infectious compartment as a proportion of `INIT_INFECTION_DIST`.

        Raises
        ------
        RuntimeError
            if self.config.INIT_INFECTION_DIST does not exist. Usually created
            via self.load_initial_infection_dist_via_contact_matrix()
        """
        if not hasattr(self.config, "INIT_INFECTION_DIST"):
            raise RuntimeError(
                "this function requires `self.config.INIT_INFECTION_DIST`"
                "set if via load_initial_infection_dist_via_contact_matrix()"
                "before calling this function"
            )
        # use relative wait times in each compartment to get distribution
        #  of infections across infected vs exposed compartments
        exposed_to_infectous_ratio = self.config.EXPOSED_TO_INFECTIOUS / (
            self.config.EXPOSED_TO_INFECTIOUS + self.config.INFECTIOUS_PERIOD
        )
        return (
            1 - exposed_to_infectous_ratio
        ) * self.config.INIT_INFECTION_DIST

    def get_initial_exposed_distribution(self) -> np.ndarray:
        """Get exposed proportion of initial infections.

        Returns
        -------
        np.ndarray
            actively exposed compartment as a proportion of `INIT_INFECTION_DIST`.

        Raises
        ------
        RuntimeError
            if self.config.INIT_INFECTION_DIST does not exist. Usually created
            via self.load_initial_infection_dist_via_contact_matrix()

        Notes
        -----
        Ratio of initial infections across the E and I compartments
        dictated by the ratio of their waiting times.
        ```
        self.config.EXPOSED_TO_INFECTIOUS
        / (self.config.EXPOSED_TO_INFECTIOUS + self.config.INFECTIOUS_PERIOD)
        ```
        """
        if not hasattr(self.config, "INIT_INFECTION_DIST"):
            raise RuntimeError(
                "this function requires `self.config.INIT_INFECTION_DIST`"
                "set if via load_initial_infection_dist_via_contact_matrix()"
                "before calling this function"
            )
        # use relative wait times in each compartment to get distribution
        # of infections across infected vs exposed compartments
        exposed_to_infectous_ratio = self.config.EXPOSED_TO_INFECTIOUS / (
            self.config.EXPOSED_TO_INFECTIOUS + self.config.INFECTIOUS_PERIOD
        )
        return exposed_to_infectous_ratio * self.config.INIT_INFECTION_DIST

    def load_contact_matrix(self) -> np.ndarray:
        """Load the region specific contact matrix.

        Usually sourced from https://github.com/mobs-lab/mixing-patterns

        Returns
        -------
        numpy.ndarray
            a matrix of shape (self.config.NUM_AGE_GROUPS, self.config.NUM_AGE_GROUPS)
            where `CONTACT_MATRIX[i][j]` refers to the per capita
            interaction rate between age bin `i` and `j`
        """
        return utils.load_demographic_data(
            self.config.DEMOGRAPHIC_DATA_PATH,
            self.config.REGIONS,
            self.config.NUM_AGE_GROUPS,
            self.config.AGE_LIMITS[0],
            self.config.AGE_LIMITS,
        )[self.config.REGIONS[0]]["avg_CM"]

    def load_cross_immunity_matrix(self) -> Array:
        """Load the crossimmunity matrix given the strain interactions matrix.

        Returns
        -------
        jax.Array
            matrix of shape (self.config.NUM_STRAINS, self.config.NUM_PREV_INF_HIST)
            containing the relative immune escape values for each challenging
            strain compared to each prior immune history in the model.

        Notes
        -----
        Strain interactions matrix is a matrix of shape
        (self.config.NUM_STRAINS, self.config.NUM_STRAINS)
        representing the relative immune escape risk of those who are being
        challenged by a strain in dim 0 but have recovered
        previously from a strain in dim 1.

        Neither the strain interactions matrix
        nor the crossimmunity matrix take into account waning.
        """
        return utils.strain_interaction_to_cross_immunity(
            self.config.NUM_STRAINS, self.config.STRAIN_INTERACTIONS
        )
