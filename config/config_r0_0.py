import jax.numpy as jnp

from config.config_base import ConfigBase


class ConfigScenario(ConfigBase):
    """
    This is a Config file for a particular scenario,
    in which we want to test a 3 strain model with inital R0 of 0 for each strain, and default vaccination.
    Through inheritance this class will inherit all non-listed parameters from ConfigBase, and can even add its own!
    """

    def __init__(self) -> None:
        self.SCENARIO_NAME = (
            "R0 = 0 for all strains, with vaccination scenario"
        )
        # set scenario parameters here
        self.STRAIN_SPECIFIC_R0 = jnp.array([0, 0, 0])
        # pass all modified scenario params to the base constructor to set the others.
        super().__init__(**self.__dict__)
        self.assert_valid_values()

    def assert_valid_values(self):
        super().assert_valid_values()
        assert True, "any new parameters should be tested here"
