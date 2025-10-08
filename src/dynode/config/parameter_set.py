from typing import Any

import numpyro  # type: ignore
import numpyro.distributions as Dist  # type: ignore
from pydantic import (
    BaseModel,
    ConfigDict,
)

from .deterministic_parameter import DeterministicParameter


class ParameterSet(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    def sample_distributions(self):
        obj_dict = self.model_dump()

        for key, value in obj_dict.items():
            if issubclass(type(value), Dist.Distribution):
                numpyro.sample(value)

    # Need to support more than one parameter set for this function
    # the dependent parameters could be in multiple other sets
    def resolve_deterministic(self, dependent_parameter_set: dict[str, Any]):
        obj_dict = self.model_dump()

        for key, value in obj_dict.items():
            if isinstance(value, DeterministicParameter):
                numpyro.determinisitc(value.resolve(dependent_parameter_set))
