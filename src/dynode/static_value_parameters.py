"""
This class is responsible for providing parameters to the model in the case that
no parameters are being sampled, and thus no complex inference or fitting is needed.
"""

from . import SEIC_Compartments, utils
from .config import Config
from .parameters import Parameters


class StaticValueParameters:
    """A Parameters class made for use on all static parameters, with no in-built sampling mechanism"""

    def __init__(
        self,
        INITIAL_STATE: SEIC_Compartments,
        parameters: Parameters,
        runner_config_path: str,
        global_variables_path: str,
    ) -> None:
        runner_json = open(runner_config_path, "r").read()
        global_json = open(global_variables_path, "r").read()
        self.config = Config(global_json).add_file(runner_json)
        self.INITIAL_STATE = INITIAL_STATE
        self.config.POPULATION = utils.retrieve_population_counts(
            self.INITIAL_STATE
        )
        self.parameters = parameters
        # load self.config.VACCINATION_MODEL_KNOTS/
        # VACCINATION_MODEL_KNOT_LOCATIONS/VACCINATION_MODEL_BASE_EQUATIONS
        self.load_vaccination_model()
        # load self.config.CONTACT_MATRIX
        self.load_contact_matrix()
        # rest of the work is handled by the AbstractParameters
