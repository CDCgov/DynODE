from pydantic import (
    BaseModel,
    ConfigDict,
)

from .parameter_set import ParameterSet
from .params import ParameterWrapper

# from .sample import sample_then_resolve
from .simulation_config import SimulationConfig


class CompartmentalModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    parameters: ParameterWrapper
    data: ParameterSet
    config: SimulationConfig

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
        data = self.data.model_copy(deep=True)
        config = self.config.model_copy(deep=True)

        distributions = parameters.distributions
        deterministic_params = parameters.deterministic_params
        transmission_params = parameters.transmission_params

        distributions.sample_distributions()
        deterministic_params.resolve_deterministic(distributions)

        for strain in transmission_params.strains:
            strain.sample_distributions()
            strain.resolve_deterministic(distributions)

        transmission_params.build_interaction_matrix()

        return parameters, data, config
