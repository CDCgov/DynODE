from __future__ import annotations

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

    `name` is the logical parameter key used in parameter contexts. `site_name`
    and runtime scope control the NumPyro sample-site name. This distinction is
    required for repeated model instances such as multi-year hierarchical
    experiments, where each year may have a logical `H1_r0` parameter but a
    unique NumPyro site like `2015_H1_r0`.
    """

    model_config = ConfigDict(
        extra="forbid", frozen=True, arbitrary_types_allowed=True
    )

    name: DynodeName = Field(description="Logical sampled-parameter name.")
    distribution: PriorDistributionSpec = Field(
        description="Prior distribution spec."
    )

    site_name: str | None = Field(
        default=None,
        description="Optional explicit NumPyro sample-site name.",
    )
    scope: str | None = Field(
        default=None,
        description="Optional default scope used when constructing the site name.",
    )

    expand_shape: tuple[int, ...] = Field(default_factory=tuple)
    event_dim: NonNegativeInt = 0
    description: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)

    @property
    def dependencies(self) -> set[str]:
        return self.distribution.dependencies()

    @property
    def data_dependencies(self) -> set[str]:
        data_deps = getattr(self.distribution, "data_dependencies", None)
        if callable(data_deps):
            return set(data_deps())
        return set(data_deps or set())

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

    def sample_site_name(self, scope: str | None = None) -> str:
        if self.site_name is not None:
            return self.site_name
        resolved_scope = scope if scope is not None else self.scope
        if resolved_scope:
            return f"{resolved_scope}_{self.name}"
        return str(self.name)

    def to_numpyro(
        self,
        context: dict[str, Any] | None = None,
        data: Any | None = None,
    ) -> dist.Distribution:
        distribution = self.distribution.to_numpyro(context=context, data=data)
        if self.expand_shape:
            distribution = distribution.expand(self.expand_shape)
        if self.event_dim:
            distribution = distribution.to_event(self.event_dim)
        return distribution
