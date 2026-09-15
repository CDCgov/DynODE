from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpyro.distributions as dist

from .base import DistributionSpec

if TYPE_CHECKING:
    from .unions import DistributionTransformSpec, PriorDistributionSpec


class TransformedDistributionSpec(DistributionSpec):
    type: Literal["transformed"] = "transformed"

    base: PriorDistributionSpec
    transforms: tuple[DistributionTransformSpec, ...]

    def dependencies(self) -> set[str]:
        deps = self.base.dependencies()
        for transform in self.transforms:
            deps |= transform.dependencies()
        return deps

    def to_numpyro(
        self, context: dict[str, Any] | None = None, data: Any | None = None
    ) -> dist.Distribution:
        return dist.TransformedDistribution(
            self.base.to_numpyro(context=context, data=data),
            [
                transform.to_numpyro(context=context, data=data)
                for transform in self.transforms
            ],
        )
