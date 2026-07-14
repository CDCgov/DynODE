from __future__ import annotations

from itertools import combinations
from typing import Any, Literal

from pydantic import Field, model_validator
from typing_extensions import Self

from dynode.structure.bins.base import BinSpec
from dynode.typing import DynodeName

from .base import DimensionSpec


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
    """

    type: Literal["immune_history_full"] = "immune_history_full"

    bins: tuple[BinSpec, ...] = Field(
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
    def _build_bins(strain_names: tuple[str, ...]) -> tuple[BinSpec, ...]:
        all_immune_histories: list[BinSpec] = [BinSpec(name="none")]

        for history_size in range(1, len(strain_names) + 1):
            for combo in combinations(strain_names, history_size):
                all_immune_histories.append(BinSpec(name="_".join(combo)))

        return tuple(all_immune_histories)

    @classmethod
    def _coerce_strain_names_from_data(
        cls, data: dict[str, Any]
    ) -> tuple[str, ...]:
        if "strain_names" in data:
            return tuple(data["strain_names"])

        if "strains" in data:
            return tuple(cls._extract_strain_names(data["strains"]))

        return tuple()


class LastStrainImmuneHistoryDimension(ImmuneHistoryDimension):
    """
    Immune-history dimension that tracks only the most recent infection.
    """

    type: Literal["immune_history_last"] = "immune_history_last"

    bins: tuple[BinSpec, ...] = Field(
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
    def _build_bins(strain_names: tuple[str, ...]) -> tuple[BinSpec, ...]:
        return tuple(
            [BinSpec(name="none")]
            + [BinSpec(name=strain_name) for strain_name in strain_names]
        )

    @classmethod
    def _coerce_strain_names_from_data(
        cls, data: dict[str, Any]
    ) -> tuple[str, ...]:
        if "strain_names" in data:
            return tuple(data["strain_names"])

        if "strains" in data:
            return tuple(cls._extract_strain_names(data["strains"]))

        return tuple()
