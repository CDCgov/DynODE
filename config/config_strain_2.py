import jax.numpy as jnp

from config.config_base import ConfigBase


class ConfigScenario(ConfigBase):
    """
    This is a Config file for a particular scenario,
    in which we want to test a 2 strain model with all default params.
    Through inheritance this class will inherit all non-listed parameters from ConfigBase, and can even add its own!
    """

    def __init__(self) -> None:
        self.SCENARIO_NAME = "NEW SCENARIO"
        # set scenario parameters here
        self.STRAIN_SPECIFIC_R0 = jnp.array([1.5, 1.5])
        self.NUM_STRAINS = 2
        self.STRAIN_INTERACTIONS = jnp.array(
            [
                [
                    1.0,
                    0.7,
                ],  # delta
                [
                    0.7,
                    1.0,
                ],  # omi
            ]
        )
        self.VAX_EFF_MATRIX = jnp.array(
            [
                [0, 0.34, 0.68],  # delta
                [0, 0.24, 0.48],  # omicron1
            ]
        )
        # pass all modified scenario params to the base constructor to set the others.
        super().__init__(**self.__dict__)

    def assert_valid_values(self):
        super().assert_valid_values()
        assert True, "any new parameters should be tested here"
