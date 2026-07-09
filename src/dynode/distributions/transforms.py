from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dynode.value.coercion import coerce_value_fields
from dynode.value.constant import ConstantValueSpec

if TYPE_CHECKING:
    from dynode.value.unions import DistributionValue


class AffineTransformSpec(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, arbitrary_types_allowed=True
    )

    type: Literal["affine"] = "affine"
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

    def dependencies(self) -> set[str]:
        return self.loc.dependencies() | self.scale.dependencies()

    def to_numpyro(
        self, context: dict[str, Any] | None = None, data: Any | None = None
    ):
        import numpyro.distributions.transforms as transforms

        return transforms.AffineTransform(
            loc=self.loc.evaluate(context=context, data=data),
            scale=self.scale.evaluate(context=context, data=data),
        )
