"""Provides static parameters to ODEs to solve."""

from . import SEIC_Compartments, logger
from .abstract_parameters import AbstractParameters
from .config import Config


class StaticValueParameters(AbstractParameters):
    """A Parameters class made for use on static parameters, with no sampling mechanism."""

    def __init__(
        self,
        INITIAL_STATE: SEIC_Compartments,
        runner_config_path: str,
        global_variables_path: str,
    ) -> None:
        """Initialize an parameters object with config JSONS and an initial state.

        Parameters
        ----------
        global_variables_path : str
            Path to global JSON for parameters shared across all components
            of the model.
        distributions_path : str
            Path to runner specific JSON of parameters containing static parameters.
        runner : MechanisticRunner
            Runner class to solve ODEs and return infection timeseries.
        initial_state : SEIC_Compartments
            Initial compartment state at t=0.
        """
        runner_json = open(runner_config_path, "r").read()
        global_json = open(global_variables_path, "r").read()
        self.config = Config(global_json).add_file(runner_json)
        self.INITIAL_STATE = INITIAL_STATE
        # load self.config.POPULATION
        self.config.POPULATION = self.retrieve_population_counts()
        # load self.config.VACCINATION_MODEL_KNOTS/
        # VACCINATION_MODEL_KNOT_LOCATIONS/VACCINATION_MODEL_BASE_EQUATIONS
        # load all vaccination splines
        (
            self.config.VACCINATION_MODEL_KNOTS,
            self.config.VACCINATION_MODEL_KNOT_LOCATIONS,
            self.config.VACCINATION_MODEL_BASE_EQUATIONS,
        ) = self.load_vaccination_model()
        self.config.CONTACT_MATRIX = self.load_contact_matrix()
