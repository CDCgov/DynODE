from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpyro.distributions as dist
from pydantic import Field, model_validator
from typing_extensions import Self

from dynode.value.coercion import coerce_value_fields
from dynode.value.constant import ConstantValueSpec

from .base import DistributionSpec
from .validators import (
    _validate_constant_numeric,
    _validate_constant_positive,
)

if TYPE_CHECKING:
    from dynode.value.unions import DistributionValue


class LogNormalSpec(DistributionSpec):
    type: Literal["lognormal"] = "lognormal"

    loc: DistributionValue = Field(
        default_factory=lambda: ConstantValueSpec(value=0.0)
    )
    scale: DistributionValue = Field(
        default_factory=lambda: ConstantValueSpec(value=1.0)
    )

    @model_validator(mode="before")
    @classmethod
    def coerce_values(cls, data: Any) -> Any:
        return coerce_value_fields(data, ("loc", "scale"))

    @model_validator(mode="after")
    def validate_parameters(self) -> Self:
        _validate_constant_numeric(self.loc, "LogNormal loc")
        _validate_constant_positive(self.scale, "LogNormal scale")
        return self

    def dependencies(self) -> set[str]:
        return self.loc.dependencies() | self.scale.dependencies()

    def to_numpyro(
        self,
        context: dict[str, Any] | None = None,
        data: Any | None = None,
    ) -> dist.Distribution:
        return dist.LogNormal(
            loc=self.loc.evaluate(context=context, data=data),
            scale=self.scale.evaluate(context=context, data=data),
        )
