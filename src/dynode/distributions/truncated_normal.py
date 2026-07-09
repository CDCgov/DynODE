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
    _validate_constant_positive,
)

if TYPE_CHECKING:
    from dynode.value.unions import DistributionValue


class TruncatedNormalSpec(DistributionSpec):
    type: Literal["truncated_normal"] = "truncated_normal"

    loc: DistributionValue = Field(
        default_factory=lambda: ConstantValueSpec(value=0.0)
    )
    scale: DistributionValue = Field(
        default_factory=lambda: ConstantValueSpec(value=1.0)
    )
    low: DistributionValue | None = None
    high: DistributionValue | None = None

    @model_validator(mode="before")
    @classmethod
    def coerce_values(cls, data: Any) -> Any:
        return coerce_value_fields(
            data,
            ("loc", "scale", "low", "high"),
        )

    @model_validator(mode="after")
    def validate_parameters(self) -> Self:
        _validate_constant_numeric(
            self.loc,
            "TruncatedNormal loc",
        )
        _validate_constant_positive(
            self.scale,
            "TruncatedNormal scale",
        )

        if self.low is not None:
            _validate_constant_numeric(
                self.low,
                "TruncatedNormal low",
            )

        if self.high is not None:
            _validate_constant_numeric(
                self.high,
                "TruncatedNormal high",
            )

        _validate_constant_bounds(
            self.low,
            self.high,
        )

        return self

    def dependencies(self) -> set[str]:
        deps = self.loc.dependencies() | self.scale.dependencies()

        if self.low is not None:
            deps |= self.low.dependencies()

        if self.high is not None:
            deps |= self.high.dependencies()

        return deps

    def to_numpyro(
        self,
        context: dict[str, Any] | None = None,
        data: Any | None = None,
    ) -> dist.Distribution:
        return dist.TruncatedNormal(
            loc=self.loc.evaluate(context=context, data=data),
            scale=self.scale.evaluate(context=context, data=data),
            low=None
            if self.low is None
            else self.low.evaluate(context=context, data=data),
            high=None
            if self.high is None
            else self.high.evaluate(context=context, data=data),
        )
