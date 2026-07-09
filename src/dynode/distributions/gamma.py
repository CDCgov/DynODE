from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpyro.distributions as dist
from pydantic import model_validator
from typing_extensions import Self

from dynode.value.coercion import coerce_value_fields

from .base import DistributionSpec
from .validators import _validate_constant_positive

if TYPE_CHECKING:
    from dynode.value.unions import DistributionValue


class GammaSpec(DistributionSpec):
    type: Literal["gamma"] = "gamma"

    concentration: DistributionValue
    rate: DistributionValue

    @model_validator(mode="before")
    @classmethod
    def coerce_values(cls, data: Any) -> Any:
        return coerce_value_fields(data, ("concentration", "rate"))

    @model_validator(mode="after")
    def validate_parameters(self) -> Self:
        _validate_constant_positive(
            self.concentration,
            "Gamma concentration",
        )
        _validate_constant_positive(
            self.rate,
            "Gamma rate",
        )
        return self

    def dependencies(self) -> set[str]:
        return self.concentration.dependencies() | self.rate.dependencies()

    def to_numpyro(
        self,
        context: dict[str, Any] | None = None,
        data: Any | None = None,
    ) -> dist.Distribution:
        return dist.Gamma(
            concentration=self.concentration.evaluate(
                context=context, data=data
            ),
            rate=self.rate.evaluate(context=context, data=data),
        )
