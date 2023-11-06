import jax.numpy as jnp

from config.config_base import ConfigBase


class ConfigScenario(ConfigBase):
    """
    This is an example Config file for a particular scenario,
    in which we want to test a 2 strain model with inital R0 of 1.5 for eachs train, and no vaccination.
    Through inheritance this class will inherit all non-listed parameters from ConfigBase, and can even add its own!
    """

    def __init__(self) -> None:
        self.SCENARIO_NAME = "EXAMPLE 3 AGE BIN, 2 STRAIN, NO VAX SCENARIO"
        # example needed to run a model with only two strains, each with inital R0 of 1.5, and no vaccination.
        self.MINIMUM_AGE = 0
        # changing age limits, this impacts NUM_AGE_GROUPS when super().__init__() is called
        self.AGE_LIMITS = [self.MINIMUM_AGE, 18, 65]
        self.NUM_STRAINS = 2
        self.STRAIN_SPECIFIC_R0 = jnp.array([1.5, 1.5])
        self.VACCINATION_RATE = 0
        self.UNIQUE_PARAM_TO_THIS_SCENARIO = "Example"
        # pass all modified scenario params to the base constructor to set the others.
        super().__init__(**self.__dict__)

    def assert_valid_values(self):
        super().assert_valid_values()
        assert self.UNIQUE_PARAM_TO_THIS_SCENARIO, "new param checks!"
