from __future__ import annotations

from typing import Annotated

from pydantic import Field

from .beta import BetaSpec
from .exponential import ExponentialSpec
from .gamma import GammaSpec
from .half_normal import HalfNormalSpec
from .lognormal import LogNormalSpec
from .normal import NormalSpec
from .registered import RegisteredDistributionSpec
from .transformed import TransformedDistributionSpec
from .transforms import AffineTransformSpec
from .truncated_normal import TruncatedNormalSpec
from .uniform import UniformSpec

DistributionTransformSpec = Annotated[
    AffineTransformSpec,
    Field(discriminator="type"),
]


PriorDistributionSpec = Annotated[
    NormalSpec
    | LogNormalSpec
    | GammaSpec
    | ExponentialSpec
    | BetaSpec
    | HalfNormalSpec
    | TruncatedNormalSpec
    | UniformSpec
    | TransformedDistributionSpec
    | RegisteredDistributionSpec,
    Field(discriminator="type"),
]


TransformedDistributionSpec.model_rebuild(
    _types_namespace={
        "PriorDistributionSpec": PriorDistributionSpec,
        "DistributionTransformSpec": DistributionTransformSpec,
    }
)
