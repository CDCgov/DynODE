import jax.numpy as jnp

from config.config_base import ConfigBase


class ConfigScenario(ConfigBase):
    """
    This is a Config file for a particular scenario,
    in which we want to test a 2 strain model with inital R0 of 1.5 for eachs train, and no vaccination.
    Through inheritance this class will inherit all non-listed parameters from ConfigBase, and can even add its own!
    """

    def __init__(self) -> None:
        self.SCENARIO_NAME = "r0 = 1.5, No vaccination, 2 strain model"
        # set scenario parameters here
        self.NUM_STRAINS = 2
        self.STRAIN_SPECIFIC_R0 = jnp.array([1.5, 1.5])
        self.VACCINATION_RATE = 0
        # pass all modified scenario params to the base constructor to set the others.
        super().__init__(**self.__dict__)

    def assert_valid_values(self):
        super().assert_valid_values()
        assert True, "any new parameters should be tested here"
