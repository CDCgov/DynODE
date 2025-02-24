"""Dimension types for ODE compartments."""

from itertools import combinations
from typing import List

from pydantic import BaseModel, Field, field_validator
from typing_extensions import Self

from .bins import Bin, DiscretizedPositiveIntBin
from .strains import Strain


class Dimension(BaseModel):
    """A dimension of an compartment."""

    name: str = Field(
        description="""Dimension name, must be unique within a Compartment"""
    )
    bins: List[Bin] = Field(
        description="""Bins/cells within this dimension."""
    )

    def __len__(self):
        """Get len of a Dimension."""
        return len(self.bins)

    @field_validator("bins", mode="after")
    @classmethod
    def _check_bins_same_type(cls, bins) -> Self:
        """Assert all bins are of same type and bins is not empty."""
        assert len(bins) > 0, "can not have dimension with no bins"
        bin_type = type(bins[0])
        assert all([type(b) is bin_type for b in bins]), (
            "can not instantiate dimension with mixed type bins. Found list of types %s"
            % str([type(b) for b in bins])
        )
        return bins

    @field_validator("bins", mode="after")
    @classmethod
    def _check_bin_names_unique(cls, bins: list[Bin]) -> list[Bin]:
        assert len(bins) > 0, "can not have dimension with no bins"
        names = [b.name for b in bins]
        assert len(set(names)) == len(
            names
        ), "Dimension of categorical bins must have unique bin names."
        return bins

    @field_validator("bins", mode="after")
    @classmethod
    def sort_discretized_int_bins(cls, bins: list[Bin]) -> list[Bin]:
        """Assert that DiscretizedPositiveIntBin do not overlap and sorts them lowest to highest."""
        assert len(bins) > 0, "can not have dimension with no bins"
        if all(isinstance(bin, DiscretizedPositiveIntBin) for bin in bins):
            # sort age bins with lowest min_value first
            bins_sorted = sorted(
                bins, key=lambda b: b.min_value, reverse=False
            )
            # assert that bins dont overlap now they are sorted
            assert all(
                [
                    bins_sorted[i].max_value < bins[i + 1].min_value
                    for i in range(len(bins) - 1)
                ]
            ), "DiscretizedPositiveIntBin within a dimension can not overlap."
            return bins_sorted
        return bins


class VaccinationDimension(Dimension):
    """A vaccination dimension of a compartment, supporting ordinal (and optionally seasonal) vaccinations."""

    def __init__(
        self, max_ordinal_vaccinations: int, seasonal_vaccination: bool = False
    ):
        """Specify a vaccination dimension with some ordinal doses and optional seasonal dose."""
        if seasonal_vaccination:
            max_ordinal_vaccinations += 1
        bins: list[Bin] = [
            DiscretizedPositiveIntBin(min_value=vax_count, max_value=vax_count)
            for vax_count in range(max_ordinal_vaccinations + 1)
        ]
        super().__init__(name="vax", bins=bins)


class FullStratifiedImmuneHistory(Dimension):
    """A type of immune history which represents all possible unique infections."""

    def __init__(self, strains: list[Strain]) -> None:
        """Create a fully stratified immune history dimension."""
        # TODO add a no-infection bin
        strain_names = [s.strain_name for s in strains]
        all_immune_histories = [Bin(name="none")]
        for strain in range(1, len(strain_names) + 1):
            combs = combinations(strain_names, strain)
            all_immune_histories.extend(
                [Bin(name="_".join(comb)) for comb in combs]
            )

        super().__init__(name="hist", bins=all_immune_histories)


class LastStrainImmuneHistory(Dimension):
    """Immune history dimension that only tracks most recent infection."""

    def __init__(self, strains: list[Strain]) -> None:
        """Create an immune history dimension that only tracks last infected strain."""
        # TODO add a no-infection bin
        strain_names = [s.strain_name for s in strains]
        bins: list[Bin] = [Bin(name=state) for state in strain_names]
        bins.insert(0, Bin(name="none"))
        super().__init__(name="hist", bins=bins)
