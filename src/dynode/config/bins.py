"""Bin types for ODE compartment models."""

from pydantic import (
    BaseModel,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    model_validator,
)
from typing_extensions import Literal, Self

from dynode.typing import DynodeName


class Bin(BaseModel):
    """A catch-all bin class meant to represent an individual cell of an ODE compartment."""

    name: DynodeName = Field(
        description="bin name, must be unique to the dimension. Can not begin with a numeric"
    )


class DiscretizedPositiveIntBin(Bin):
    """Bin with a distinct discretized positive int inclusive min/max."""

    min_value: NonNegativeInt = Field(
        description="Minimum value of this bin, inclusive."
    )
    max_value: NonNegativeInt = Field(
        description="Maximum value of this bin, inclusive."
    )

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
            name = f"range_{min_value}_{max_value}"
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
            name of the bin, by default f"A{min_value}_{max_value}" if None
        """
        if name is None:
            name = f"a{min_value}_{max_value}"
        super().__init__(name=name, min_value=min_value, max_value=max_value)


class WaneBin(Bin):
    """Waning bin with a protection value and waning time in days."""

    waiting_time: PositiveFloat = Field(
        description="""Average time from when individual first enters
        the bin to them leaving. waning_rate = 1/waning_time = proportion of the bin
        exits per day. math.inf -> does not wane out of this bin.""",
    )
    base_protection: NonNegativeFloat = Field(
        le=1.0,
        description="""Proportion of immune protection retained by populations within
          this bin, between [0-1]. May be subsequently modified by strain immune escape.""",
    )
