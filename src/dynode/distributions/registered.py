from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpyro.distributions as dist
from pydantic import Field, model_validator

from .base import DistributionSpec

if TYPE_CHECKING:
    from dynode.value.unions import DistributionValue


class RegisteredDistributionSpec(DistributionSpec):
    type: Literal["registered"] = "registered"

    registry_key: str
    kwargs: dict[str, DistributionValue] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def coerce_kwargs(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        kwargs = data.get("kwargs")
        if isinstance(kwargs, dict):
            data["kwargs"] = {
                key: value
                if isinstance(value, dict) and "type" in value
                else {"type": "constant", "value": value}
                for key, value in kwargs.items()
            }
        return data

    def dependencies(self) -> set[str]:
        deps: set[str] = set()
        for value in self.kwargs.values():
            deps |= value.dependencies()
        return deps

    def to_numpyro(
        self, context: dict[str, Any] | None = None, data: Any | None = None
    ) -> dist.Distribution:
        registry = None
        if isinstance(data, dict):
            registry = data.get("distribution_registry")
        if registry is None:
            registry = (context or {}).get("distribution_registry")
        if registry is None or self.registry_key not in registry:
            raise KeyError(
                f"Unknown registered distribution {self.registry_key!r}."
            )
        kwargs = {
            key: value.evaluate(context=context, data=data)
            for key, value in self.kwargs.items()
        }
        return registry[self.registry_key](**kwargs)
