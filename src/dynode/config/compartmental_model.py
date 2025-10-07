from pydantic import (
    BaseModel,
    ConfigDict,
)

# from .parameter_set import ParameterSet
from .params import ParameterWrapper
from .sample import sample_then_resolve
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

    def parameter_init(self):
        distributions = sample_then_resolve(self.distributions)

        configs = {}
        for key, config in self.configs.items():
            config_copy = config.model_copy(deep=True)
            config_copy.inject_parameters(
                injection_parameter_set=distributions
            )
            config_copy.sample_then_resolve_parameters(prefix=str(key))
            configs[key] = config_copy

        return configs, distributions
