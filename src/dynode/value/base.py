from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from typing import Any

from pydantic import BaseModel, ConfigDict


class ValueSpec(BaseModel):
    """
    Base class for serializable values used across the framework.

    Used by:
    - DistributionSpec
    - DeterministicSpec
    - InitializerSpec
    - InteractionSpec
    - StrainSpec
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )

    type: str

    def dependencies(self) -> set[str]:
        """
        All parameter-like dependencies.

        This includes sampled parameter refs and deterministic refs.
        """
        return (
            self.parameter_dependencies() | self.deterministic_dependencies()
        )

    def parameter_dependencies(self) -> set[str]:
        return set()

    def deterministic_dependencies(self) -> set[str]:
        return set()

    def data_dependencies(self) -> set[str]:
        return set()

    def evaluate(
        self,
        context: Mapping[str, Any] | None = None,
        data: Any | None = None,
    ) -> Any:
        raise NotImplementedError
