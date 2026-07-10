from .age import AgeDimensionSpec
from .base import DimensionSpec
from .immune_history import (
    FullStratifiedImmuneHistoryDimension,
    ImmuneHistoryDimension,
    LastStrainImmuneHistoryDimension,
)
from .unions import AnyDimensionSpec
from .vaccination import VaccinationDimensionSpec
from .wane import WaneDimensionSpec

__all__ = [
    "AgeDimensionSpec",
    "AnyDimensionSpec",
    "DimensionSpec",
    "FullStratifiedImmuneHistoryDimension",
    "ImmuneHistoryDimension",
    "LastStrainImmuneHistoryDimension",
    "VaccinationDimensionSpec",
    "WaneDimensionSpec",
]
