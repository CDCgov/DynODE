from __future__ import annotations

from typing import Any, ClassVar, Literal

from pydantic import Field, NonNegativeInt, model_validator
from typing_extensions import Self

from .base import BinSpec


class DiscretizedPositiveIntBin(BinSpec):
    """
    Bin with an inclusive non-negative integer range.

    The range is inclusive:

        min_value <= x <= max_value
    """

    type: Literal["discretized_positive_int"] = "discretized_positive_int"

    default_name_prefix: ClassVar[str] = "range"

    min_value: NonNegativeInt = Field(
        description="Minimum integer value in this bin, inclusive.",
    )

    max_value: NonNegativeInt = Field(
        description="Maximum integer value in this bin, inclusive.",
    )

    @model_validator(mode="before")
    @classmethod
    def fill_default_name(cls, data: Any) -> Any:
        """
        Allows compact construction without an explicit name.
        """
        if not isinstance(data, dict):
            return data

        data = dict(data)

        if data.get("name") is not None:
            return data

        if "min_value" in data and "max_value" in data:
            data["name"] = cls.default_name(
                min_value=data["min_value"],
                max_value=data["max_value"],
            )

        return data

    @model_validator(mode="after")
    def validate_range(self) -> Self:
        if self.min_value > self.max_value:
            raise ValueError(
                "DiscretizedPositiveIntBin requires min_value <= max_value. "
                f"Got min_value={self.min_value}, max_value={self.max_value}."
            )

        return self

    @classmethod
    def default_name(
        cls,
        min_value: int,
        max_value: int,
    ) -> str:
        return f"{cls.default_name_prefix}_{min_value}_{max_value}"

    @property
    def width(self) -> int:
        """
        Number of integer values contained by the bin.
        """
        return self.max_value - self.min_value + 1

    @property
    def bounds(self) -> tuple[int, int]:
        return self.min_value, self.max_value

    def contains(self, value: Any) -> bool:
        if not isinstance(value, int):
            return False

        return self.min_value <= value <= self.max_value

    def overlaps(self, other: DiscretizedPositiveIntBin) -> bool:
        return not (
            self.max_value < other.min_value
            or other.max_value < self.min_value
        )

    def is_adjacent_before(self, other: DiscretizedPositiveIntBin) -> bool:
        return self.max_value + 1 == other.min_value

    def gap_before(self, other: DiscretizedPositiveIntBin) -> int:
        """
        Number of missing integer values between this bin and another bin.

        Returns 0 if the bins are adjacent or overlapping.
        """
        gap = other.min_value - self.max_value - 1
        return max(gap, 0)
