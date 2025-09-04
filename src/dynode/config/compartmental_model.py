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

    def model_post_init(self, __context) -> None:
        self.shared_parameters = sample_then_resolve(self.shared_parameters)

        for _, config in self.configs.items():
            config.inject_parameters(parameter_set=self.shared_parameters)
            config.sample_then_resolve_parameters()

    def numpyro_model(self, **kwargs):
        """User must implement this method to define the NumPyro model."""

        raise NotImplementedError(
            "implement functionality to get initial state"
        )
