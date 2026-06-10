from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict


class DistributionSpec(BaseModel, ABC):
    model_config = ConfigDict(extra="forbid", frozen=True)
    type: str

    @abstractmethod
    def to_numpyro(self):
        raise NotImplementedError
