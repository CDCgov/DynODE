from pydantic import (
    BaseModel,
    ConfigDict,
)

from .parameter_set import ParameterSet
from .sample import sample_then_resolve
from .simulation_config import SimulationConfig


class CompartmentalModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    shared_parameters: ParameterSet  # add Pydantic Field to class attributes
    configs: dict[int, SimulationConfig]

    def numpyro_model(self, **kwargs):
        """User must implement this method to define the NumPyro model."""

        raise NotImplementedError(
            "implement functionality to get initial state"
        )

    def parameter_init(self):
        shared_parameters = sample_then_resolve(self.shared_parameters)

        configs = {}
        for key, config in self.configs.items():
            config_copy = config.model_copy(deep=True)
            config_copy.inject_parameters(
                injection_parameter_set=shared_parameters
            )
            config_copy.sample_then_resolve_parameters(prefix=str(key))
            configs[key] = config_copy

        return configs, shared_parameters
