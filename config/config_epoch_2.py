import jax.numpy as jnp

from config.config_base import ConfigBase


class ConfigEpoch(ConfigBase):
    """
    This is an example Config file for a particular scenario,
    in which we want to test a 2 strain model with inital R0 of 1.5 for eachs train, and no vaccination.
    Through inheritance this class will inherit all non-listed parameters from ConfigBase, and can even add its own!
    """

    def __init__(self) -> None:
        self.SCENARIO_NAME = "Epoch 2, omicron, BA2, XBB"
        # set scenario parameters here
        self.STRAIN_SPECIFIC_R0 = jnp.array([1.8, 3.0, 3.0])  # R0s
        self.STRAIN_INTERACTIONS = jnp.array(
            [
                [1.0, 0.7, 0.49],  # omicron
                [0.7, 1.0, 0.7],  # BA2
                [0.49, 0.7, 1.0],  # XBB
            ]
        )
        self.VAX_EFF_MATRIX = jnp.array(
            [
                [0, 0.34, 0.68],  # omicron
                [0, 0.24, 0.48],  # BA2
                [0, 0.14, 0.28],  # XBB
            ]
        )
        self.all_strains_supported = [
            "wildtype",
            "alpha",
            "delta",
            "omicron",
            "BA2/BA5",
            "XBB1.5",
        ]
        # specifies the number of days after the model INIT date this epoch occurs
        self.DAYS_AFTER_INIT_DATE = 250
        # DO NOT CHANGE THE FOLLOWING TWO LINES
        super().__init__(**self.__dict__)
        # Do not add any scenario parameters below, may create inconsistent state

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
