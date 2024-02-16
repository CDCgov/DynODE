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
        self.config = Config(global_variables_path).add_file(
            config_initializer_path
        )
        # self.__dict__.update(**config.__dict__)

        if not hasattr(self.config, "INITIAL_POPULATION_FRACTIONS"):
            self.load_initial_population_fractions()

        self.config.POPULATION = (
            self.config.POP_SIZE * self.config.INITIAL_POPULATION_FRACTIONS
        )
        # self.POPULATION.shape = (NUM_AGE_GROUPS,)

        if not hasattr(self.config, "INIT_IMMUNE_HISTORY"):
            self.load_immune_history_via_abm()
        # self.INIT_IMMUNE_HISTORY.shape = (age, hist, num_vax, waning)

        # stratify initial infections appropriately across age, hist, vax counts
        if not (
            hasattr(self.config, "INIT_INFECTED_DIST")
            and hasattr(self.config, "INIT_EXPOSED_DIST")
        ):
            self.load_init_infection_infected_and_exposed_dist_via_abm()

        # load initial state using INIT_IMMUNE_HISTORY, INIT_INFECTED_DIST, and INIT_EXPOSED_DIST
        self.INITIAL_STATE = self.load_initial_state(
            self.config.INITIAL_INFECTIONS
        )

    def load_immune_history_via_abm(self):
        self.config.INIT_IMMUNE_HISTORY = utils.past_immune_dist_from_abm(
            self.config.SIM_DATA_PATH,
            self.config.NUM_AGE_GROUPS,
            self.config.AGE_LIMITS,
            self.config.MAX_VAX_COUNT,
            self.config.WANING_TIMES,
            self.config.NUM_WANING_COMPARTMENTS,
            self.config.NUM_STRAINS,
            self.config.STRAIN_IDX,
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
        `self.config.INIT_INFECTION_DIST`: jnp.array(int)
            populates values using abm to produce a distribution of how new infections are
            stratified by age bin, vax, immune_history, and strain strata. All new infections are classified in STRAIN_IDX.omicron
        `self.config.INIT_EXPOSED_DIST`: jnp.array(int)
            proportion of INIT_INFECTION_DIST that falls into exposed compartment,
            stratified by age bin, vax, immune_history, and strain strata. All new exposures are classified in STRAIN_IDX.omicron
        `self.config.INIT_INFECTED_DIST`: jnp.array(int)
            proportion of INIT_INFECTION_DIST that falls into infected compartment,
            stratified by age bin, vax, immune_history, and strain strata. All new infected are classified in STRAIN_IDX.omicron
        `self.INITIAL_INFECTIONS`: float
            if `INITIAL_INFECTIONS` is not specified in the config, will use the proportion of the total population
            that is exposed or infected in the abm, multiplied by the population size, for the models number of infections.
        """
        (
            self.config.INIT_INFECTION_DIST,
            self.config.INIT_EXPOSED_DIST,
            self.config.INIT_INFECTED_DIST,
            proportion_infected,
        ) = utils.init_infections_from_abm(
            self.config.SIM_DATA_PATH,
            self.config.NUM_AGE_GROUPS,
            self.config.AGE_LIMITS,
            self.config.MAX_VAX_COUNT,
            self.config.WANING_TIMES,
            self.config.NUM_STRAINS,
            self.config.STRAIN_IDX,
        )
        if self.config.INITIAL_INFECTIONS is None:
            self.config.INITIAL_INFECTIONS = (
                self.config.POP_SIZE * proportion_infected
            )

    def load_initial_state(self, initial_infections: float):
        """
        a function which takes a number of initial infections,
        disperses them across infectious and exposed compartments according to the INIT_INFECTED_DIST
        and INIT_EXPOSED_DIST distributions, then subtracts both those populations from the total population and
        places the remaining individuals in the susceptible compartment,
        distributed according to the INIT_IMMUNE_HISTORY distribution.

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
        initial_infectious_count = (
            initial_infections * self.config.INIT_INFECTED_DIST
        )
        initial_infectious_count_ages = jnp.sum(
            initial_infectious_count,
            axis=(
                self.config.I_AXIS_IDX.hist,
                self.config.I_AXIS_IDX.vax,
                self.config.I_AXIS_IDX.strain,
            ),
        )
        # create population distribution using INIT_EXPOSED_DIST, then sum them for later use
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
        # susceptible / partial susceptible = Total population - infected_count - exposed_count
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
