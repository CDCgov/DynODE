"""Bin types for ODE compartment models."""

from pydantic import (
    BaseModel,
    NonNegativeInt,
    PositiveFloat,
    model_validator,
)
from typing_extensions import Self


class Bin(BaseModel):
    """A catch-all bin class meant to represent an individual cell of an ODE compartment."""

    pass


class CategoricalBin(Bin):
    """Bin with a distinct name."""

    name: str


class DiscretizedPositiveIntBin(Bin):
    """Bin with a distinct discretized positive int inclusive min/max."""

    min_value: NonNegativeInt
    max_value: NonNegativeInt

    @model_validator(mode="after")
    def bin_valid_side(self) -> Self:
        """Assert that min_value <= max_value."""
        assert self.min_value <= self.max_value
        return self


class AgeBin(DiscretizedPositiveIntBin):
    """Age bin with inclusive mix and max age values."""

    pass


class WaneBin(CategoricalBin):
    """Waning bin with a protection value and waning time in days."""

    waning_time: NonNegativeInt
    waning_protection: PositiveFloat
