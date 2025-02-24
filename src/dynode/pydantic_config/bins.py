"""Bin types for ODE compartment models."""

from pydantic import (
    BaseModel,
    NonNegativeFloat,
    NonNegativeInt,
    model_validator,
)
from typing_extensions import Self


class Bin(BaseModel):
    """A catch-all bin class meant to represent an individual cell of an ODE compartment."""

    name: str


class DiscretizedPositiveIntBin(Bin):
    """Bin with a distinct discretized positive int inclusive min/max."""

    min_value: NonNegativeInt
    max_value: NonNegativeInt

    def __init__(self, min_value, max_value, name=None):
        """Initialize a Discretized bin with inclusive min/max and sensible default name.

        Parameters
        ----------
        min_value : int
            minimum value contained by the bin (inclusive)
        max_value : int
            maximum value contained by the bin (inclusive)
        name : str, optional
            name of the bin, by default f"{min_value}_{max_value}" if None
        """
        if name is None:
            name = f"{min_value}_{max_value}"
        super().__init__(name=name, min_value=min_value, max_value=max_value)

    @model_validator(mode="after")
    def _bin_valid_side(self) -> Self:
        """Assert that min_value <= max_value."""
        assert self.min_value <= self.max_value
        return self


class AgeBin(DiscretizedPositiveIntBin):
    """Age bin with inclusive mix and max age values, fills in name of bin for you."""

    def __init__(self, min_value, max_value, name=None):
        """Initialize a Discretized bin with inclusive min/max and sensible default name.

        Parameters
        ----------
        min_value : int
            minimum value contained by the bin (inclusive)
        max_value : int
            maximum value contained by the bin (inclusive)
        name : str, optional
            name of the bin, by default f"{min_value}_{max_value}" if None
        """
        super().__init__(name=name, min_value=min_value, max_value=max_value)


class WaneBin(Bin):
    """Waning bin with a protection value and waning time in days."""

    waning_time: NonNegativeInt
    waning_protection: NonNegativeFloat
