from __future__ import annotations

from itertools import combinations
from math import isinf
from types import SimpleNamespace
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveFloat,
    field_validator,
    model_validator,
)
from typing_extensions import Self

from dynode.typing import DynodeName, UnitIntervalFloat

from .bin_spec import AnyBinSpec, coerce_bin_specs

class DimensionSpec(BaseModel):
    """
    Declarative dimension specification for an ODE compartment.

    A dimension represents one axis of stratification within a compartment.

    Examples
    --------
    Age dimension:

        DimensionSpec(
            name="age",
            bins=(AgeBin(...), AgeBin(...)),
        )

    Immune-history dimension:

        LastStrainImmuneHistoryDimension(
            strain_names=("delta", "omicron"),
        )

    This class should not:
    - know about compartments
    - know about parameters
    - know about observed data
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
    def validate_bins(cls, bins: tuple[Bin, ...]) -> tuple[Bin, ...]:
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
        return {
            bin_.name: i
            for i, bin_ in enumerate(self.bins)
        }

    @property
    def idx(self) -> SimpleNamespace:
        """
        Enum-like helper for bin indexes.

        Example
        -------
        dimension.idx.age_0_4
        dimension.idx.naive
        dimension.idx.delta
        """
        namespace = SimpleNamespace()

        for bin_idx, bin_ in enumerate(self.bins):
            setattr(namespace, bin_.name, bin_idx)

        return namespace

    def get_bin(self, name: str) -> Bin:
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
    def _validate_bins_not_empty(bins: tuple[Bin, ...]) -> None:
        if not bins:
            raise ValueError("DimensionSpec.bins must contain at least one bin.")

    @staticmethod
    def _validate_bin_names_unique(bins: tuple[Bin, ...]) -> None:
        names = [bin_.name for bin_ in bins]
        duplicates = sorted({name for name in names if names.count(name) > 1})

        if duplicates:
            raise ValueError(
                f"Dimension bin names must be unique. Duplicates: {duplicates}."
            )

    @staticmethod
    def _validate_bins_same_type(bins: tuple[Bin, ...]) -> None:
        """
        Keep the old behavior: a single dimension should not mix bin classes.

        If you later have a valid use case for mixed bins, relax this method or
        make it configurable.
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
    def _all_discretized_integer_bins(bins: tuple[Bin, ...]) -> bool:
        return all(
            isinstance(bin_, DiscretizedPositiveIntBin)
            for bin_ in bins
        )

    @classmethod
    def _validate_discretized_integer_bins_sorted(
        cls,
        bins: tuple[Bin, ...],
    ) -> None:
        if not cls._all_discretized_integer_bins(bins):
            return

        sorted_bins = tuple(
            sorted(
                bins,
                key=lambda bin_: bin_.min_value,
            )
        )

        if bins != sorted_bins:
            raise ValueError(
                "Dimensions made of DiscretizedPositiveIntBin must be sorted "
                f"by min_value. Got: {bins}."
            )

    @classmethod
    def _validate_discretized_integer_bins_do_not_overlap(
        cls,
        bins: tuple[Bin, ...],
    ) -> None:
        if not cls._all_discretized_integer_bins(bins):
            return

        for current, next_ in zip(bins, bins[1:]):
            if current.max_value >= next_.min_value:
                raise ValueError(
                    "DiscretizedPositiveIntBin values cannot overlap within a "
                    f"dimension. Found overlap between {current} and {next_}."
                )

    @classmethod
    def _validate_discretized_integer_bins_have_no_gaps(
        cls,
        bins: tuple[Bin, ...],
    ) -> None:
        if not cls._all_discretized_integer_bins(bins):
            return

        for current, next_ in zip(bins, bins[1:]):
            expected_next_min = current.max_value + 1

            if expected_next_min != next_.min_value:
                raise ValueError(
                    "Dimensions containing DiscretizedPositiveIntBin cannot "
                    "have gaps. "
                    f"Expected next min_value={expected_next_min}, but got "
                    f"{next_.min_value}. Gap found between {current} and {next_}."
                )

class AgeDimensionSpec(DimensionSpec):
    """
    Dimension for age stratification.

    This is a thin specialization of DimensionSpec that requires AgeBin bins.
    """

    type: Literal["age"] = "age"

    name: DynodeName = Field(
        default="age",
        description="Age dimension name.",
    )

    bins: tuple[AgeBin, ...] = Field(
        min_length=1,
        description="Age bins.",
    )

class VaccinationDimensionSpec(DimensionSpec):
    """
    Vaccination dimension.

    Tracks ordinal vaccine dose count, optionally including one seasonal dose.

    Example
    -------
    VaccinationDimensionSpec(
        max_ordinal_vaccinations=3,
        seasonal_vaccination=True,
    )

    This creates bins:

        v0, v1, v2, v3, v4

    where v4 represents the additional seasonal dose.
    """

    type: Literal["vaccination"] = "vaccination"

    name: DynodeName = Field(
        default="vax",
        description="Vaccination dimension name.",
    )

    max_ordinal_vaccinations: NonNegativeInt = Field(
        description="Maximum tracked ordinal vaccination count, excluding seasonal vaccination.",
    )

    seasonal_vaccination: bool = Field(
        default=False,
        description="Whether this dimension tracks an additional seasonal vaccination.",
    )

    bins: tuple[DiscretizedPositiveIntBin, ...] = Field(
        default_factory=tuple,
        description="Vaccination count bins. Usually generated automatically.",
    )

    @model_validator(mode="before")
    @classmethod
    def build_bins_if_missing(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        data = dict(data)

        if data.get("bins"):
            return data

        if "max_ordinal_vaccinations" not in data:
            return data

        max_ordinal_vaccinations = int(data["max_ordinal_vaccinations"])
        seasonal_vaccination = bool(data.get("seasonal_vaccination", False))

        max_tracked_vaccinations = max_ordinal_vaccinations

        if seasonal_vaccination:
            max_tracked_vaccinations += 1

        data["bins"] = tuple(
            DiscretizedPositiveIntBin(
                name=f"v{vax_count}",
                min_value=vax_count,
                max_value=vax_count,
            )
            for vax_count in range(max_tracked_vaccinations + 1)
        )

        return data

    @property
    def max_shots(self) -> int:
        """
        Maximum number of tracked vaccinations in the dimension.

        Additional shots do not increase the count.
        """
        return len(self.bins) - 1

class ImmuneHistoryDimension(DimensionSpec):
    """
    Marker base class for immune-history dimensions.

    ModelSpec can use isinstance(..., ImmuneHistoryDimension) to find and
    validate immune-history dimensions against TransmissionSpec.strains.
    """

    name: DynodeName = Field(
        default="hist",
        description="Immune-history dimension name.",
    )

    strain_names: tuple[DynodeName, ...] = Field(
        min_length=1,
        description="Strain names used to construct the immune-history bins.",
    )

    def validate_against_strains(self, strains: Any) -> None:
        """
        Validate this immune-history dimension against model strains.

        Accepts either:
        - StrainSpec objects with .name
        - dicts with ['name']
        - plain strings

        This keeps dimension.py from importing StrainSpec.
        """
        actual_names = tuple(self._extract_strain_names(strains))

        if actual_names != self.strain_names:
            raise ValueError(
                f"Immune-history dimension {self.name!r} was built from "
                f"strain_names={self.strain_names}, but model transmission "
                f"uses strain_names={actual_names}."
            )

        expected = self.rebuild_from_strains(actual_names)

        if expected.bins != self.bins:
            raise ValueError(
                f"Immune-history dimension {self.name!r} bins do not match "
                "the bins expected from the model strains. "
                f"Expected {expected.bin_names}, found {self.bin_names}."
            )

    @classmethod
    def _extract_strain_names(cls, strains: Any) -> list[str]:
        names: list[str] = []

        for strain in strains:
            if isinstance(strain, str):
                names.append(strain)
                continue

            if isinstance(strain, dict):
                names.append(strain["name"])
                continue

            names.append(strain.name)

        return names

    @classmethod
    def rebuild_from_strains(
        cls,
        strain_names: tuple[str, ...],
    ) -> ImmuneHistoryDimension:
        raise NotImplementedError

class FullStratifiedImmuneHistoryDimension(ImmuneHistoryDimension):
    """
    Immune-history dimension that tracks all unique combinations of prior infection.

    For N strains, this creates 2^N bins:

        none
        strain_1
        strain_2
        ...
        strain_1_strain_2
        ...
    """

    type: Literal["immune_history_full"] = "immune_history_full"

    bins: tuple[Bin, ...] = Field(
        default_factory=tuple,
        description="Immune-history bins. Usually generated automatically.",
    )

    @model_validator(mode="before")
    @classmethod
    def build_bins_if_missing(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        data = dict(data)

        if data.get("bins"):
            return data

        strain_names = cls._coerce_strain_names_from_data(data)

        if not strain_names:
            return data

        data["strain_names"] = tuple(strain_names)
        data["bins"] = cls._build_bins(tuple(strain_names))

        return data

    @model_validator(mode="after")
    def validate_immune_history_bins(self) -> Self:
        expected_bins = self._build_bins(self.strain_names)

        if self.bins != expected_bins:
            raise ValueError(
                f"FullStratifiedImmuneHistoryDimension bins do not match "
                f"strain_names={self.strain_names}. "
                f"Expected {[bin_.name for bin_ in expected_bins]}, "
                f"found {self.bin_names}."
            )

        return self

    @classmethod
    def rebuild_from_strains(
        cls,
        strain_names: tuple[str, ...],
    ) -> FullStratifiedImmuneHistoryDimension:
        return cls(
            strain_names=tuple(strain_names),
        )

    @staticmethod
    def _build_bins(strain_names: tuple[str, ...]) -> tuple[Bin, ...]:
        all_immune_histories: list[Bin] = [Bin(name="none")]

        for history_size in range(1, len(strain_names) + 1):
            for combo in combinations(strain_names, history_size):
                all_immune_histories.append(
                    Bin(name="_".join(combo))
                )

        return tuple(all_immune_histories)

    @classmethod
    def _coerce_strain_names_from_data(cls, data: dict[str, Any]) -> tuple[str, ...]:
        if "strain_names" in data:
            return tuple(data["strain_names"])

        if "strains" in data:
            return tuple(cls._extract_strain_names(data["strains"]))

        return tuple()

class LastStrainImmuneHistoryDimension(ImmuneHistoryDimension):
    """
    Immune-history dimension that tracks only the most recent infection.

    For N strains, this creates N + 1 bins:

        none
        strain_1
        strain_2
        ...
    """

    type: Literal["immune_history_last"] = "immune_history_last"

    bins: tuple[Bin, ...] = Field(
        default_factory=tuple,
        description="Immune-history bins. Usually generated automatically.",
    )

    @model_validator(mode="before")
    @classmethod
    def build_bins_if_missing(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        data = dict(data)

        if data.get("bins"):
            return data

        strain_names = cls._coerce_strain_names_from_data(data)

        if not strain_names:
            return data

        data["strain_names"] = tuple(strain_names)
        data["bins"] = cls._build_bins(tuple(strain_names))

        return data

    @model_validator(mode="after")
    def validate_immune_history_bins(self) -> Self:
        expected_bins = self._build_bins(self.strain_names)

        if self.bins != expected_bins:
            raise ValueError(
                f"LastStrainImmuneHistoryDimension bins do not match "
                f"strain_names={self.strain_names}. "
                f"Expected {[bin_.name for bin_ in expected_bins]}, "
                f"found {self.bin_names}."
            )

        return self

    @classmethod
    def rebuild_from_strains(
        cls,
        strain_names: tuple[str, ...],
    ) -> LastStrainImmuneHistoryDimension:
        return cls(
            strain_names=tuple(strain_names),
        )

    @staticmethod
    def _build_bins(strain_names: tuple[str, ...]) -> tuple[Bin, ...]:
        return tuple(
            [Bin(name="none")]
            + [Bin(name=strain_name) for strain_name in strain_names]
        )

    @classmethod
    def _coerce_strain_names_from_data(cls, data: dict[str, Any]) -> tuple[str, ...]:
        if "strain_names" in data:
            return tuple(data["strain_names"])

        if "strains" in data:
            return tuple(cls._extract_strain_names(data["strains"]))

        return tuple()

class WaneDimensionSpec(DimensionSpec):
    """
    Dimension for waning immunity after recovery or vaccination.

    Each bin has:
    - waiting_time
    - base_protection

    The final bin must have infinite waiting_time, because population should
    not wane out of the terminal waning state.
    """

    type: Literal["wane"] = "wane"

    name: DynodeName = Field(
        default="wane",
        description="Waning dimension name.",
    )

    waiting_times: tuple[PositiveFloat, ...] = Field(
        min_length=1,
        description="Waiting time for each waning bin.",
    )

    base_protections: tuple[UnitIntervalFloat, ...] = Field(
        min_length=1,
        description="Base protection for each waning bin.",
    )

    bins: tuple[WaneBin, ...] = Field(
        default_factory=tuple,
        description="Waning bins. Usually generated automatically.",
    )

    @model_validator(mode="before")
    @classmethod
    def build_bins_if_missing(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        data = dict(data)

        if data.get("bins"):
            return data

        waiting_times = data.get("waiting_times")
        base_protections = data.get("base_protections")

        if waiting_times is None or base_protections is None:
            return data

        if len(waiting_times) != len(base_protections):
            raise ValueError(
                "WaneDimensionSpec requires equal-length waiting_times and "
                "base_protections."
            )

        data["bins"] = tuple(
            WaneBin(
                name=f"W{idx}",
                waiting_time=wait_time,
                base_protection=base_protection,
            )
            for idx, (wait_time, base_protection) in enumerate(
                zip(waiting_times, base_protections)
            )
        )

        return data

    @model_validator(mode="after")
    def validate_wane_dimension(self) -> Self:
        if len(self.waiting_times) != len(self.base_protections):
            raise ValueError(
                "WaneDimensionSpec requires equal-length waiting_times and "
                "base_protections."
            )

        if len(self.bins) != len(self.waiting_times):
            raise ValueError(
                "WaneDimensionSpec bins, waiting_times, and base_protections "
                "must have matching lengths."
            )

        last_wane_bin = self.bins[-1]

        if not isinf(last_wane_bin.waiting_time):
            raise ValueError(
                "The final WaneBin must have infinite waiting_time."
            )

        return self

AnyDimensionSpec = Annotated[
    DimensionSpec
    | AgeDimensionSpec
    | VaccinationDimensionSpec
    | FullStratifiedImmuneHistoryDimension
    | LastStrainImmuneHistoryDimension
    | WaneDimensionSpec,
    Field(discriminator="type"),
]
