from __future__ import annotations

from math import isinf, isnan
from typing import Annotated, Literal

from pydantic import Field, model_validator
from typing_extensions import Self

from .base import BinSpec


Probability = Annotated[
    float,
    Field(ge=0.0, le=1.0),
]


class WaneBin(BinSpec):
    """
    Waning bin with a protection value and waning time.

    waiting_time is the average time spent in the bin.
    If waiting_time is math.inf, the bin is terminal.
    """

    type: Literal["wane"] = "wane"

    waiting_time: float = Field(
        gt=0.0,
        allow_inf_nan=True,
        description=(
            "Average time spent in this waning bin. "
            "math.inf means population does not wane out of this bin."
        ),
    )

    base_protection: Probability = Field(
        description=(
            "Proportion of immune protection retained by populations within "
            "this bin, between 0 and 1."
        ),
    )

    @model_validator(mode="after")
    def validate_wane_bin(self) -> Self:
        if isnan(self.waiting_time):
            raise ValueError("WaneBin.waiting_time cannot be NaN.")

        if self.waiting_time <= 0:
            raise ValueError("WaneBin.waiting_time must be positive.")

        if isnan(self.base_protection):
            raise ValueError("WaneBin.base_protection cannot be NaN.")

        return self

    @property
    def is_terminal(self) -> bool:
        """
        Whether population remains in this bin indefinitely.
        """
        return isinf(self.waiting_time)

    @property
    def waning_rate(self) -> float:
        """
        Rate at which population exits this bin.

        For terminal bins, this is 0.
        """
        if self.is_terminal:
            return 0.0

        return 1.0 / self.waiting_time
