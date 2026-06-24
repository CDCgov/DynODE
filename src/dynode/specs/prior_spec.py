from typing import Any

import numpyro.distributions as dist
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    model_validator,
)
from typing_extensions import Self

from dynode.typing import DynodeName

from .distribution_spec import PriorDistributionSpec


class PriorSpec(BaseModel):
    """
    Declarative specification for one sampled NumPyro parameter.

    This class should describe the prior, not perform sampling.
    Sampling belongs in the runtime / NumPyro layer.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )

    name: DynodeName = Field(
        description="Name of the sampled parameter. This becomes the NumPyro sample site name.",
    )

    distribution: PriorDistributionSpec = Field(
        description="Distribution specification for this sampled parameter.",
    )

    expand_shape: tuple[int, ...] = Field(
        default_factory=tuple,
        description=(
            "Optional batch shape to apply to the distribution using .expand(...). "
            "Use this for vector or matrix-valued parameters."
        ),
    )

    event_dim: NonNegativeInt = Field(
        default=0,
        description=(
            "Number of rightmost batch dimensions to reinterpret as event dimensions "
            "using .to_event(event_dim)."
        ),
    )

    description: str | None = Field(
        default=None,
        description="Optional human-readable description of the parameter.",
    )

    metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Optional metadata for documentation, auditing, or UI display.",
    )

    @property
    def dependencies(self) -> set[str]:
        """
        Parameters needed to construct this prior distribution.

        Usually empty for simple priors, but useful for hierarchical priors.
        """
        return self.distribution.dependencies()

    @model_validator(mode="after")
    def validate_shape_and_event_dim(self) -> Self:
        if self.event_dim > len(self.expand_shape):
            raise ValueError(
                "event_dim cannot be larger than the number of dimensions in "
                f"expand_shape. Got event_dim={self.event_dim}, "
                f"expand_shape={self.expand_shape}."
            )

        if any(dim <= 0 for dim in self.expand_shape):
            raise ValueError(
                f"All expand_shape dimensions must be positive. Got {self.expand_shape}."
            )

        return self

    def to_numpyro(
        self,
        context: dict[str, Any] | None = None,
    ) -> dist.Distribution:
        """
        Compile this prior spec into a NumPyro distribution.
        """
        distribution = self.distribution.to_numpyro(context)

        if self.expand_shape:
            distribution = distribution.expand(self.expand_shape)

        if self.event_dim:
            distribution = distribution.to_event(self.event_dim)

        return distribution
