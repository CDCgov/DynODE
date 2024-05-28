from config.config import Config
from mechanistic_model.abstract_parameters import AbstractParameters


class StaticValueParameters(AbstractParameters):
    def __init__(
        self, INITIAL_STATE, runner_config_path, global_variables_path
    ):
        runner_json = open(runner_config_path, "r").read()
        global_json = open(global_variables_path, "r").read()
        self.config = Config(global_json).add_file(runner_json)
        self.INITIAL_STATE = INITIAL_STATE
        self.retrieve_population_counts()
        self.load_vaccination_model()
        self.load_contact_matrix()
