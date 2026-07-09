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


class BetaSpec(DistributionSpec):
    type: Literal["beta"] = "beta"

    concentration1: DistributionValue
    concentration0: DistributionValue

    @model_validator(mode="before")
    @classmethod
    def coerce_values(cls, data: Any) -> Any:
        return coerce_value_fields(
            data,
            ("concentration1", "concentration0"),
        )

    @model_validator(mode="after")
    def validate_parameters(self) -> Self:
        _validate_constant_positive(
            self.concentration1,
            "Beta concentration1",
        )
        _validate_constant_positive(
            self.concentration0,
            "Beta concentration0",
        )
        return self

    def dependencies(self) -> set[str]:
        return (
            self.concentration1.dependencies()
            | self.concentration0.dependencies()
        )

    def to_numpyro(
        self,
        context: dict[str, Any] | None = None,
        data: Any | None = None,
    ) -> dist.Distribution:
        return dist.Beta(
            concentration1=self.concentration1.evaluate(
                context=context, data=data
            ),
            concentration0=self.concentration0.evaluate(
                context=context, data=data
            ),
        )
