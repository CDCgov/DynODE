from __future__ import annotations

from math import isinf, isnan
from typing import Annotated, Any, ClassVar, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    model_validator,
)
from typing_extensions import Self

from dynode.typing import DynodeName


class BinSpec(BaseModel):
    """
    Base declarative bin specification.

    A bin is one named cell within a DimensionSpec.

    Examples
    --------
    Generic categorical bin:

        BinSpec(name="none")

    Age bin:

        AgeBin(min_value=0, max_value=4)

    Waning bin:

        WaneBin(
            name="W0",
            waiting_time=30.0,
            base_protection=0.8,
        )

    This class should not:
    - know about dimensions
    - know about compartments
    - know about parameters
    - call NumPyro
    - build JAX arrays
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )

    type: Literal["generic"] = Field(
        default="generic",
        description="Bin type discriminator.",
    )

    name: DynodeName = Field(
        description=(
            "Bin name. Must be unique within a DimensionSpec. "
            "Should be safe to use as an attribute name."
        ),
    )

    description: str | None = Field(
        default=None,
        description="Optional human-readable description.",
    )

    metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Optional metadata for documentation, UI display, or auditing.",
    )

    def contains(self, value: Any) -> bool:
        """
        Whether this bin contains a value.

        Generic categorical bins only match by name.
        Numeric subclasses override this method.
        """
        return value == self.name


class DiscretizedPositiveIntBin(BinSpec):
    """
    Bin with an inclusive non-negative integer range.

    Useful for dimensions such as:
    - age
    - vaccination count
    - number of prior infections
    - ordinal severity levels

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
        Allows compact construction:

            DiscretizedPositiveIntBin(min_value=0, max_value=4)

        instead of:

            DiscretizedPositiveIntBin(
                name="range_0_4",
                min_value=0,
                max_value=4,
            )
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


class AgeBin(DiscretizedPositiveIntBin):
    """
    Age bin with inclusive minimum and maximum ages.

    Examples
    --------
    AgeBin(min_value=0, max_value=4)
        name defaults to "a0_4"

    AgeBin(name="age_0_4", min_value=0, max_value=4)
        explicit name
    """

    type: Literal["age"] = "age"

    default_name_prefix: ClassVar[str] = "a"

    @classmethod
    def default_name(
        cls,
        min_value: int,
        max_value: int,
    ) -> str:
        return f"a{min_value}_{max_value}"

    @property
    def min_age(self) -> int:
        return self.min_value

    @property
    def max_age(self) -> int:
        return self.max_value

    @property
    def label(self) -> str:
        return f"{self.min_value}-{self.max_value}"

    def contains_age(self, age: int) -> bool:
        return self.contains(age)


Probability = Annotated[
    float,
    Field(ge=0.0, le=1.0),
]


class WaneBin(BinSpec):
    """
    Waning bin with a protection value and waning time.

    waiting_time is the average time spent in the bin.

        waning_rate = 1 / waiting_time

    If waiting_time is math.inf, then the bin is terminal and population does
    not wane out of it.
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
            "this bin, between 0 and 1. This may later be modified by "
            "strain-specific immune escape."
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


AnyBinSpec = Annotated[
    BinSpec | DiscretizedPositiveIntBin | AgeBin | WaneBin,
    Field(discriminator="type"),
]


def as_bin_spec(value: Any) -> Any:
    """
    Coerce compact bin inputs into explicit bin specs.

    Useful before Pydantic discriminated-union parsing.

    Examples
    --------
    {"name": "none"}
        -> {"type": "generic", "name": "none"}

    "none"
        -> {"type": "generic", "name": "none"}

    {"type": "age", "min_value": 0, "max_value": 4}
        -> unchanged
    """
    if isinstance(value, BinSpec):
        return value

    if isinstance(value, str):
        return {
            "type": "generic",
            "name": value,
        }

    if isinstance(value, dict):
        value = dict(value)

        if "type" in value:
            return value

        if "min_value" in value and "max_value" in value:
            return {
                "type": "discretized_positive_int",
                **value,
            }

        if "name" in value:
            return {
                "type": "generic",
                **value,
            }

    return value


def coerce_bin_specs(values: Any) -> Any:
    """
    Coerce a list/tuple of compact bin configs into explicit bin specs.
    """
    if values is None:
        return values

    return tuple(as_bin_spec(value) for value in values)
