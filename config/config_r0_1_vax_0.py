import jax.numpy as jnp

from config.config_base import ConfigBase


class ConfigScenario(ConfigBase):
    """
    This is a Config file for a particular scenario,
    in which we want to test a 3 strain model with inital R0 of 1.0 for each strain, and no vaccination.
    Through inheritance this class will inherit all non-listed parameters from ConfigBase, and can even add its own!
    """

    def __init__(self) -> None:
        self.SCENARIO_NAME = "R0 = 1, No Vaccination Scenario"
        # set scenario parameters here
        self.STRAIN_SPECIFIC_R0 = jnp.array([1.0, 1.0, 1.0])
        self.VACCINATION_RATE = 0
        # pass all modified scenario params to the base constructor to set the others.
        super().__init__(**self.__dict__)

    def assert_valid_values(self):
        super().assert_valid_values()
        assert True, "any new parameters should be tested here"
