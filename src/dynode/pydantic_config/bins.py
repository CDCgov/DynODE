"""Bin types for ODE compartment models."""

from typing import Optional

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

    def __eq__(self, value):
        if type(self) is type(value):
            return self.__dict__ == value.__dict__
        else:
            return False


class DiscretizedPositiveIntBin(Bin):
    """Bin with a distinct discretized positive int inclusive min/max."""

    min_value: NonNegativeInt
    max_value: NonNegativeInt
    name: Optional[str] = None

    def __init__(self, min_value, max_value, name=None):
        if name is None:
            name = f"{min_value}_{max_value}"
        super().__init__(name=name, min_value=min_value, max_value=max_value)

    @model_validator(mode="after")
    def bin_valid_side(self) -> Self:
        """Assert that min_value <= max_value."""
        assert self.min_value <= self.max_value
        return self


class AgeBin(DiscretizedPositiveIntBin):
    """Age bin with inclusive mix and max age values, fills in name of bin for you."""

    pass


class WaneBin(Bin):
    """Waning bin with a protection value and waning time in days."""

    waning_time: NonNegativeInt
    waning_protection: NonNegativeFloat
