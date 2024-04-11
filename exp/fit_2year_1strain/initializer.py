import jax.numpy as jnp

from config.config import Config
from mechanistic_model.abstract_initializer import MechanisticInitializer


class EarlyCovidInitializer(MechanisticInitializer):
    def __init__(
        self, config_initializer_path, global_variables_path, initial_sero
    ):
        """
        An intializer for COVID-19 model that does not rely on ABM output. The initializer make some
        basic assumptions to the initial immune history based on input initial seroprevalence. These
        assumptions include: 1. Only used for 1 strain model, 2. Applicable to early pandemic thus
        80% of past infections went to first waning bin and 20% went to second waning bin, 3. 1/3 of
        initial infections are exposed and 2/3 are infectious.
        """
        initializer_json = open(config_initializer_path, "r").read()
        global_json = open(global_variables_path, "r").read()
        self.config = Config(global_json).add_file(initializer_json)

        self.load_initial_population_fractions()
        self.config.POPULATION = (
            self.config.POP_SIZE * self.config.INITIAL_POPULATION_FRACTIONS
        )
        # self.POPULATION.shape = (NUM_AGE_GROUPS,)

        self.load_immune_history(initial_sero)
        # self.INIT_IMMUNE_HISTORY.shape = (age, hist, num_vax, waning)

        # stratify initial infections appropriately across age, hist, vax counts
        self.load_init_infection_dist()

        # load initial state using INIT_IMMUNE_HISTORY, INIT_INFECTED_DIST, and INIT_EXPOSED_DIST
        self.INITIAL_STATE = self.load_initial_state(
            self.config.INITIAL_INFECTIONS
        )

    def load_immune_history(self, initial_sero):
        arr = jnp.zeros(
            (
                self.config.NUM_AGE_GROUPS,
                2**self.config.NUM_STRAINS,
                self.config.MAX_VAX_COUNT + 1,
                self.config.NUM_WANING_COMPARTMENTS,
            )
        )

        for i, s in enumerate(initial_sero):
            arr = arr.at[i, 0, 0, self.config.NUM_WANING_COMPARTMENTS - 1].set(
                1 - s
            )
            arr = arr.at[i, 1, 0, 0:2].set([s * 0.8, s * 0.2])

        self.config.INIT_IMMUNE_HISTORY = arr

    def load_init_infection_dist(
        self,
        ei_split=[0.34, 0.66],
        age_split=jnp.array([0.41, 0.33, 0.23, 0.04]),
    ):
        arr_e = jnp.zeros(
            (
                self.config.NUM_AGE_GROUPS,
                2**self.config.NUM_STRAINS,
                self.config.MAX_VAX_COUNT + 1,
                self.config.NUM_STRAINS,
            )
        )
        arr_i = jnp.zeros(
            (
                self.config.NUM_AGE_GROUPS,
                2**self.config.NUM_STRAINS,
                self.config.MAX_VAX_COUNT + 1,
                self.config.NUM_STRAINS,
            )
        )

        arr_e = arr_e.at[:, 0, 0, 0].set(ei_split[0] * age_split)
        arr_i = arr_i.at[:, 0, 0, 0].set(ei_split[1] * age_split)

        self.config.INIT_EXPOSED_DIST = arr_e
        self.config.INIT_INFECTED_DIST = arr_i

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
