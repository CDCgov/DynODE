from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from dynode.structure.bins.base import BinSpec
from dynode.structure.bins.coercion import coerce_bin_specs
from dynode.structure.bins.discretized import DiscretizedPositiveIntBin
from dynode.structure.bins.unions import AnyBinSpec
from dynode.typing import DynodeName


class DimensionSpec(BaseModel):
    """
    Declarative dimension specification for an ODE compartment.

    A dimension represents one axis of stratification within a compartment.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )

    type: Literal["generic"] = Field(
        default="generic",
        description="Dimension type discriminator.",
    )

    name: DynodeName = Field(
        description="Dimension name. Must be unique within a CompartmentSpec.",
    )

    bins: tuple[AnyBinSpec, ...] = Field(
        min_length=1,
        description="Bins/cells within this dimension.",
    )

    description: str | None = Field(
        default=None,
        description="Optional human-readable description.",
    )

    metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Optional metadata for documentation, UI display, or auditing.",
    )

    @model_validator(mode="before")
    @classmethod
    def coerce_bins(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        data = dict(data)

        if "bins" in data:
            data["bins"] = coerce_bin_specs(data["bins"])

        return data

    @field_validator("bins")
    @classmethod
    def validate_bins(
        cls, bins: tuple[AnyBinSpec, ...]
    ) -> tuple[AnyBinSpec, ...]:
        cls._validate_bins_not_empty(bins)
        cls._validate_bin_names_unique(bins)
        cls._validate_bins_same_type(bins)
        cls._validate_discretized_integer_bins_sorted(bins)
        cls._validate_discretized_integer_bins_do_not_overlap(bins)
        cls._validate_discretized_integer_bins_have_no_gaps(bins)
        return bins

    def __len__(self) -> int:
        return len(self.bins)

    @property
    def bin_names(self) -> list[str]:
        return [bin_.name for bin_ in self.bins]

    @property
    def bins_to_idx(self) -> dict[str, int]:
        return {bin_.name: i for i, bin_ in enumerate(self.bins)}

    @property
    def idx(self) -> SimpleNamespace:
        """
        Enum-like helper for bin indexes.
        """
        namespace = SimpleNamespace()

        for bin_idx, bin_ in enumerate(self.bins):
            setattr(namespace, bin_.name, bin_idx)

        return namespace

    def get_bin(self, name: str) -> BinSpec:
        for bin_ in self.bins:
            if bin_.name == name:
                return bin_

        raise KeyError(
            f"Unknown bin {name!r} in dimension {self.name!r}. "
            f"Known bins are: {self.bin_names}."
        )

    def has_bin(self, name: str) -> bool:
        return name in self.bins_to_idx

    def index_of(self, bin_name: str) -> int:
        try:
            return self.bins_to_idx[bin_name]
        except KeyError as exc:
            raise KeyError(
                f"Unknown bin {bin_name!r} in dimension {self.name!r}. "
                f"Known bins are: {self.bin_names}."
            ) from exc

    def same_bins_as(self, other: DimensionSpec) -> bool:
        return self.bins == other.bins

    def same_layout_as(
        self,
        other: DimensionSpec,
        *,
        require_same_name: bool = True,
    ) -> bool:
        if require_same_name and self.name != other.name:
            return False

        return self.bins == other.bins

    @staticmethod
    def _validate_bins_not_empty(bins: tuple[BinSpec, ...]) -> None:
        if not bins:
            raise ValueError(
                "DimensionSpec.bins must contain at least one bin."
            )

    @staticmethod
    def _validate_bin_names_unique(bins: tuple[BinSpec, ...]) -> None:
        names = [bin_.name for bin_ in bins]
        duplicates = sorted({name for name in names if names.count(name) > 1})

        if duplicates:
            raise ValueError(
                f"Dimension bin names must be unique. Duplicates: {duplicates}."
            )

    @staticmethod
    def _validate_bins_same_type(bins: tuple[BinSpec, ...]) -> None:
        """
        Keep the old behavior: a single dimension should not mix bin classes.
        """
        first_type = type(bins[0])
        mismatched = [
            type(bin_).__name__
            for bin_ in bins
            if type(bin_) is not first_type
        ]

        if mismatched:
            raise ValueError(
                "DimensionSpec cannot contain mixed bin types. "
                f"Expected all {first_type.__name__}; got "
                f"{[type(bin_).__name__ for bin_ in bins]}."
            )

    @staticmethod
    def _as_discretized_integer_bins(
        bins: tuple[BinSpec, ...],
    ) -> tuple[DiscretizedPositiveIntBin, ...] | None:
        if not all(
            isinstance(bin_, DiscretizedPositiveIntBin) for bin_ in bins
        ):
            return None

        return tuple(bin_ for bin_ in bins)

    @classmethod
    def _validate_discretized_integer_bins_sorted(
        cls,
        bins: tuple[BinSpec, ...],
    ) -> None:
        discretized_bins = cls._as_discretized_integer_bins(bins)

        if discretized_bins is None:
            return

        sorted_bins = tuple(
            sorted(
                discretized_bins,
                key=lambda bin_: bin_.min_value,
            )
        )

        if discretized_bins != sorted_bins:
            raise ValueError(
                "Dimensions made of DiscretizedPositiveIntBin must be sorted "
                f"by min_value. Got: {discretized_bins}."
            )

    @classmethod
    def _validate_discretized_integer_bins_do_not_overlap(
        cls,
        bins: tuple[BinSpec, ...],
    ) -> None:
        discretized_bins = cls._as_discretized_integer_bins(bins)

        if discretized_bins is None:
            return

        for current, next_ in zip(discretized_bins, discretized_bins[1:]):
            if current.max_value >= next_.min_value:
                raise ValueError(
                    "DiscretizedPositiveIntBin values cannot overlap within a "
                    f"dimension. Found overlap between {current} and {next_}."
                )

    @classmethod
    def _validate_discretized_integer_bins_have_no_gaps(
        cls,
        bins: tuple[BinSpec, ...],
    ) -> None:
        discretized_bins = cls._as_discretized_integer_bins(bins)

        if discretized_bins is None:
            return

        for current, next_ in zip(discretized_bins, discretized_bins[1:]):
            expected_next_min = current.max_value + 1

            if expected_next_min != next_.min_value:
                raise ValueError(
                    "Dimensions containing DiscretizedPositiveIntBin cannot "
                    "have gaps. "
                    f"Expected next min_value={expected_next_min}, but got "
                    f"{next_.min_value}. Gap found between {current} and {next_}."
                )


DimensionSpec.model_rebuild(
    _types_namespace={
        "AnyBinSpec": AnyBinSpec,
    }
)
