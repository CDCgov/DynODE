import jax.numpy as jnp

from config.config_base import ConfigBase


class ConfigScenario(ConfigBase):
    """
    This is an example Config file for a particular scenario,
    in which we want to test a 2 strain model with inital R0 of 1.5 for eachs train, and no vaccination.
    Through inheritance this class will inherit all non-listed parameters from ConfigBase, and can even add its own!
    """

    def __init__(self) -> None:
        self.SCENARIO_NAME = "All persons begin in the recovered compartment, no one anywhere else."
        # set scenario parameters here
        # pass all modified scenario params to the base constructor to set the others.
        # DO NOT CHANGE THE FOLLOWING TWO LINES
        self.INITIAL_INFECTIONS = 0
        self.VACCINATION_RATE = 0
        super().__init__(**self.__dict__)
        # Do not add any scenario parameters below, may create inconsistent state
        self.INIT_EXPOSED_DIST = jnp.zeros(
            (self.NUM_AGE_GROUPS, self.NUM_STRAINS)
        )
        self.INIT_WANING_DIST = jnp.zeros(
            (
                self.NUM_AGE_GROUPS,
                self.NUM_STRAINS,
                self.NUM_WANING_COMPARTMENTS,
            )
        )
        # since we cant be bothered to check the serology for the age distributions of initial recovered
        # lets just define the whole population to be recovered, and equally distributed across ages.
        self.INIT_RECOVERED_DIST = jnp.array(
            [[0, 0, 1.0], [0, 0, 1.0], [0, 0, 1.0], [0, 0, 1.0]]
        )
        self.INIT_INFECTED_DIST = jnp.zeros(
            (self.NUM_AGE_GROUPS, self.NUM_STRAINS)
        )
        self.INIT_INFECTION_DIST = jnp.zeros((self.NUM_AGE_GROUPS,))

    def assert_valid_values(self):
        """
        a function designed to be called after all parameters are initalized, does a series of reasonable checks
        to ensure values are within expected ranges and no parameters directly contradict eachother.

        Raises
        ----------
        Assert Error:
            if user supplies invalid parameters, short description will be provided as to why the parameter is wrong.
        """
        super().assert_valid_values()
        assert True, "any new parameters should be tested here"
