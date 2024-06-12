"""
This class is responsible for providing parameters to the model in the case that
no parameters are being sampled, and thus no complex inference or fitting is needed.
"""

import jax

from config.config import Config
from mechanistic_model.abstract_parameters import AbstractParameters


class StaticValueParameters(AbstractParameters):
    def __init__(
        self,
        INITIAL_STATE: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
        runner_config_path: str,
        global_variables_path: str,
    ) -> None:
        runner_json = open(runner_config_path, "r").read()
        global_json = open(global_variables_path, "r").read()
        self.config = Config(global_json).add_file(runner_json)
        self.INITIAL_STATE = INITIAL_STATE
        # load self.config.POPULATION
        self.retrieve_population_counts()
        # load self.config.VACCINATION_MODEL_KNOTS/
        # VACCINATION_MODEL_KNOT_LOCATIONS/VACCINATION_MODEL_BASE_EQUATIONS
        self.load_vaccination_model()
        # load self.config.CONTACT_MATRIX
        self.load_contact_matrix()
        # rest of the work is handled by the AbstractParameters
