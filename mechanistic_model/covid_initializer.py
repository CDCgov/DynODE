import os

import jax.numpy as jnp

import utils
from config.config import Config
from mechanistic_model.abstract_initializer import MechanisticInitializer


class CovidInitializer(MechanisticInitializer):
    def __init__(self, config_initializer_path, global_variables_path):
        """
        initialize a basic abstract mechanistic model for covid19 case prediction.
        Should not be constructed directly, use build_basic_mechanistic_model() with a config file
        """
        config = Config(global_variables_path).add_file(
            config_initializer_path
        )
        self.__dict__.update(**config.__dict__)

        if not hasattr(self, "INITIAL_POPULATION_FRACTIONS"):
            self.load_initial_population_fractions()

        self.POPULATION = self.POP_SIZE * self.INITIAL_POPULATION_FRACTIONS
        # self.POPULATION.shape = (NUM_AGE_GROUPS,)

        if not hasattr(self, "INIT_IMMUNE_HISTORY"):
            self.load_immune_history_via_abm()
        # self.INIT_IMMUNE_HISTORY.shape = (age, hist, num_vax, waning)

        # stratify initial infections appropriately across age, hist, vax counts
        if not (
            hasattr(self, "INIT_INFECTED_DIST")
            and hasattr(self, "INIT_EXPOSED_DIST")
        ):
            self.load_init_infection_infected_and_exposed_dist_via_abm()

        # load initial state using INIT_IMMUNE_HISTORY, INIT_INFECTED_DIST, and INIT_EXPOSED_DIST
        self.INITIAL_STATE = self.load_initial_state(self.INITIAL_INFECTIONS)

    def load_initial_population_fractions(self):
        """
        a wrapper function which loads age demographics for the US and sets the inital population fraction by age bin.

        Updates
        ----------
        `self.INITIAL_POPULATION_FRACTIONS` : numpy.ndarray
            proportion of the total population that falls into each age group,
            length of this array is equal the number of age groups and will sum to 1.0.
        """
        populations_path = (
            self.DEMOGRAPHIC_DATA_PATH
            + "population_rescaled_age_distributions/"
        )
        self.INITIAL_POPULATION_FRACTIONS = utils.load_age_demographics(
            populations_path, self.REGIONS, self.AGE_LIMITS
        )["United States"]

    def load_immune_history_via_abm(self):
        self.INIT_IMMUNE_HISTORY = utils.past_immune_dist_from_abm(
            self.SIM_DATA_PATH,
            self.NUM_AGE_GROUPS,
            self.AGE_LIMITS,
            self.MAX_VAX_COUNT,
            self.WANING_TIMES,
            self.NUM_WANING_COMPARTMENTS,
            self.NUM_STRAINS,
            self.STRAIN_IDX,
        )

    def load_init_infection_infected_and_exposed_dist_via_abm(self):
        """
        loads the inital infection distribution by age, then separates infections into an
        infected and exposed distributions, to account for people who may not be infectious yet,
        but are part of the initial infections of the model all infections assumed to be omicron.
        utilizes the ratio between gamma and sigma to determine what proportion of inital infections belong in the
        exposed (soon to be infectious), and the already infectious compartments.

        given that `INIT_INFECTION_DIST` = `INIT_EXPOSED_DIST` + `INIT_INFECTED_DIST`

        Updates
        ----------
        `self.INIT_INFECTION_DIST`: jnp.array(int)
            populates values using abm to produce a distribution of how new infections are
            stratified by age bin, vax, immune_history, and strain strata. All new infections are classified in STRAIN_IDX.omicron
        `self.INIT_EXPOSED_DIST`: jnp.array(int)
            proportion of INIT_INFECTION_DIST that falls into exposed compartment,
            stratified by age bin, vax, immune_history, and strain strata. All new exposures are classified in STRAIN_IDX.omicron
        `self.INIT_INFECTED_DIST`: jnp.array(int)
            proportion of INIT_INFECTION_DIST that falls into infected compartment,
            stratified by age bin, vax, immune_history, and strain strata. All new infected are classified in STRAIN_IDX.omicron
        `self.INITIAL_INFECTIONS`: float
            if `INITIAL_INFECTIONS` is not specified in the config, will use the proportion of the total population
            that is exposed or infected in the abm, multiplied by the population size, for the models number of infections.
        """
        (
            self.INIT_INFECTION_DIST,
            self.INIT_EXPOSED_DIST,
            self.INIT_INFECTED_DIST,
            proportion_infected,
        ) = utils.init_infections_from_abm(
            self.SIM_DATA_PATH,
            self.NUM_AGE_GROUPS,
            self.AGE_LIMITS,
            self.MAX_VAX_COUNT,
            self.WANING_TIMES,
            self.NUM_STRAINS,
            self.STRAIN_IDX,
        )
        if self.INITIAL_INFECTIONS is None:
            self.INITIAL_INFECTIONS = self.POP_SIZE * proportion_infected

    def load_initial_state(self, initial_infections: float):
        """
        a function which takes a number of initial infections, disperses them across infectious and exposed compartments
        according to the INIT_INFECTED_DIST and INIT_EXPOSED_DIST distributions, then subtracts both those populations from the total population and
        places the remaining individuals in the susceptible compartment, distributed according to the INIT_IMMUNE_HISTORY distribution.

        Parameters
        ----------
        initial_infections: the number of infections to disperse between infectious and exposed compartments.

        Requires
        ----------
        the following variables be loaded into self:
        INIT_INFECTED_DIST: loaded in config or via load_init_infection_infected_and_exposed_dist_via_abm()
        INIT_EXPOSED_DIST: loaded in config or via load_init_infection_infected_and_exposed_dist_via_abm()
        INIT_IMMUNE_HISTORY: loaded in config or via load_immune_history_via_abm().

        Returns
        ----------
        INITIAL_STATE: tuple(jnp.ndarray)
            a tuple of len 4 representing the S, E, I, and C compartment population counts after model initialization.
        """
        # create population distribution using INIT_INFECTED_DIST, then sum them for later use
        initial_infectious_count = initial_infections * self.INIT_INFECTED_DIST
        initial_infectious_count_ages = jnp.sum(
            initial_infectious_count,
            axis=(
                self.I_AXIS_IDX.hist,
                self.I_AXIS_IDX.vax,
                self.I_AXIS_IDX.strain,
            ),
        )
        # create population distribution using INIT_EXPOSED_DIST, then sum them for later use
        initial_exposed_count = initial_infections * self.INIT_EXPOSED_DIST
        initial_exposed_count_ages = jnp.sum(
            initial_exposed_count,
            axis=(
                self.I_AXIS_IDX.hist,
                self.I_AXIS_IDX.vax,
                self.I_AXIS_IDX.strain,
            ),
        )
        # suseptible / partial susceptible = Total population - infected_count - exposed_count
        initial_suseptible_count = (
            self.POPULATION
            - initial_infectious_count_ages
            - initial_exposed_count_ages
        )[:, jnp.newaxis, jnp.newaxis, jnp.newaxis] * self.INIT_IMMUNE_HISTORY
        # cumulative count always starts at zero
        return (
            initial_suseptible_count,  # s
            initial_exposed_count,  # e
            initial_infectious_count,  # i
            jnp.zeros(initial_exposed_count.shape),  # c
        )

    def assert_valid_configuration(self):
        assert os.path.exists(self.DEMOGRAPHIC_DATA_PATH), (
            "%s is not a valid path" % self.DEMOGRAPHIC_DATA
        )
        assert os.path.exists(self.SEROLOGICAL_DATA_PATH), (
            "%s is not a valid path" % self.SEROLOGICAL_DATA
        )
        assert os.path.exists(self.SIM_DATA_PATH), (
            "%s is not a valid path" % self.SIM_DATA
        )
        assert self.MINIMUM_AGE >= 0, "no negative minimum ages, lowest is 0"
        assert (
            self.AGE_LIMITS[0] == self.MINIMUM_AGE
        ), "first age in AGE_LIMITS must be self.MINIMUM_AGE"
        assert all(
            [
                self.AGE_LIMITS[idx] > self.AGE_LIMITS[idx - 1]
                for idx in range(1, len(self.AGE_LIMITS))
            ]
        ), "AGE_LIMITS must be strictly increasing"
        assert (
            self.AGE_LIMITS[-1] < 85
        ), "age limits can not exceed 84 years of age, the last age bin is implied and does not need to be included"
        assert self.POP_SIZE > 0, "population size must be a non-zero value"
        if self.INITIAL_INFECTIONS:
            assert (
                self.INITIAL_INFECTIONS <= self.POP_SIZE
            ), "cant have more initial infections than total population size"

            assert (
                self.INITIAL_INFECTIONS >= 0
            ), "cant have negative initial infections"

        # if user has supplied custom values for distributions instead of using prebuilt ones, sanity check them here
        if self.INITIAL_POPULATION_FRACTIONS:
            assert self.INITIAL_POPULATION_FRACTIONS.shape == (
                self.NUM_AGE_GROUPS,
            ), (
                "INITIAL_POPULATION_FRACTIONS must be of shape %s, received %s"
                % (
                    str((self.NUM_AGE_GROUPS,)),
                    str(self.INITIAL_POPULATION_FRACTIONS.shape),
                )
            )
            assert (
                sum(self.INITIAL_POPULATION_FRACTIONS) == 1.0
            ), "population fractions must sum to 1"
