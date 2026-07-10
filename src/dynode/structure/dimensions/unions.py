from __future__ import annotations

from typing import Annotated

from pydantic import Field

from .age import AgeDimensionSpec
from .base import DimensionSpec
from .immune_history import (
    FullStratifiedImmuneHistoryDimension,
    LastStrainImmuneHistoryDimension,
)
from .vaccination import VaccinationDimensionSpec
from .wane import WaneDimensionSpec


AnyDimensionSpec = Annotated[
    DimensionSpec
    | AgeDimensionSpec
    | VaccinationDimensionSpec
    | FullStratifiedImmuneHistoryDimension
    | LastStrainImmuneHistoryDimension
    | WaneDimensionSpec,
    Field(discriminator="type"),
]
