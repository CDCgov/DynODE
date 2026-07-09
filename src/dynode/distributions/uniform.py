from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpyro.distributions as dist
from pydantic import Field, model_validator
from typing_extensions import Self

from dynode.value.coercion import coerce_value_fields
from dynode.value.constant import ConstantValueSpec

from .base import DistributionSpec
from .validators import (
    _validate_constant_bounds,
    _validate_constant_numeric,
)

if TYPE_CHECKING:
    from dynode.value.unions import DistributionValue


class UniformSpec(DistributionSpec):
    type: Literal["uniform"] = "uniform"

    low: DistributionValue = Field(
        default_factory=lambda: ConstantValueSpec(value=0.0)
    )
    high: DistributionValue = Field(
        default_factory=lambda: ConstantValueSpec(value=1.0)
    )

    @model_validator(mode="before")
    @classmethod
    def coerce_values(cls, data: Any) -> Any:
        return coerce_value_fields(data, ("low", "high"))

    @model_validator(mode="after")
    def validate_parameters(self) -> Self:
        _validate_constant_numeric(self.low, "Uniform low")
        _validate_constant_numeric(self.high, "Uniform high")
        _validate_constant_bounds(self.low, self.high)
        return self

    def dependencies(self) -> set[str]:
        return self.low.dependencies() | self.high.dependencies()

    def to_numpyro(
        self, context: dict[str, Any] | None = None, data: Any | None = None
    ) -> dist.Distribution:
        return dist.Uniform(
            low=self.low.evaluate(context=context, data=data),
            high=self.high.evaluate(context=context, data=data),
        )
