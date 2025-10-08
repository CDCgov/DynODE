from pydantic import (
    BaseModel,
    ConfigDict,
)

# from .parameter_set import ParameterSet
from .params import ParameterWrapper

# from .sample import sample_then_resolve
from .simulation_config import SimulationConfig


class CompartmentalModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    #    shared_parameters: ParameterSet  # add Pydantic Field to class attributes
    parameters: ParameterWrapper
    configs: dict[int, SimulationConfig]

    def numpyro_model(self, **kwargs):
        """User must implement this method to define the NumPyro model."""

        raise NotImplementedError(
            "implement functionality to get initial state"
        )

    # Still need to verify if not injecting samples into each config will even work.
    # might result in not yearly stratified results due to lack of unique site names for each year
    # however might also not be an issue since we are simulating each year seperatly. Need to verify results.
    def parameter_init(self):
        parameters = self.parameters.model_copy(deep=True)
        distributions = parameters.distributions
        deterministic_params = parameters.deterministic_params

        distributions.sample_distributions()
        deterministic_params.resolve_deterministic(distributions)

        configs = {}
        for key, config in self.configs.items():
            config_copy = config.model_copy(deep=True)
            configs[key] = config_copy

        return configs, parameters
