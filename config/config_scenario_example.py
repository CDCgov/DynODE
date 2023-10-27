import jax.numpy as jnp
from config_base import ConfigBase


class ConfigScenario(ConfigBase):
    """
    This is an example Config file for a particular scenario,
    in which we want to test a 2 strain model with inital R0 of 1.5 for eachs train, and no vaccination.
    Through inheritance this class will inherit all non-listed parameters from ConfigBase, and can even add its own!
    """

    def __init__(self) -> None:
        # load the default values then
        super().__init__()
        # example needed to run a model with only two strains, each with inital R0 of 1.5, and no vaccination.
        self.NUM_STRAINS = 2
        self.STRAIN_SPECIFIC_R0 = jnp.array([1.5, 1.5])
        self.VACCINATION_RATE = 0
        self.UNIQUE_PARAM_TO_THIS_SCENARIO = "Example"

    def assert_valid_values(self):
        super().assert_valid_values()
        assert self.UNIQUE_PARAM_TO_THIS_SCENARIO, "new param checks!"
