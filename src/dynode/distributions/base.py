from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpyro.distributions as dist
from pydantic import BaseModel, ConfigDict


class DistributionSpec(BaseModel, ABC):
    """
    Base class for declarative NumPyro distribution specs.

    Subclasses should be serializable and should not store raw NumPyro
    Distribution objects.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )

    type: str

    @abstractmethod
    def dependencies(self) -> set[str]:
        """
        Return parameter names needed to construct this distribution.
        """
        raise NotImplementedError

    @abstractmethod
    def to_numpyro(
        self,
        context: dict[str, Any] | None = None,
        data: Any | None = None,
    ) -> dist.Distribution:
        """
        Compile this spec into a NumPyro Distribution.
        """
        raise NotImplementedError
