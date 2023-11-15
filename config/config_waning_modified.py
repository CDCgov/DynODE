import jax.numpy as jnp

from config.config_base import ConfigBase


class ConfigScenario(ConfigBase):
    """
    This is an example Config file for a particular scenario,
    in which we want to test a 2 strain model with inital R0 of 1.5 for eachs train, and no vaccination.
    Through inheritance this class will inherit all non-listed parameters from ConfigBase, and can even add its own!
    """

    def __init__(self) -> None:
        self.SCENARIO_NAME = """Scenario in which waning is limited to 4 compartments,
        where each one is fit according to Ben's R script. Using dynamic waning times and fit waning protections."""
        # set scenario parameters here
        self.NUM_WANING_COMPARTMENTS = 4
        self.WANING_PROTECTIONS = jnp.array([0.48, 0.473, 0.473, 0])
        self.WANING_TIMES = [21, 142, 142, 142, 0]
        # pass all modified scenario params to the base constructor to set the others.
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
