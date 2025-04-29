"""Dimension types for ODE compartments."""

from itertools import combinations
from math import isinf
from types import SimpleNamespace
from typing import List

from pydantic import (
    BaseModel,
    Field,
    PositiveFloat,
    field_validator,
    model_validator,
)
from typing_extensions import Self

from dynode.typing import DynodeName
from dynode.typing.typing import UnitIntervalFloat

from .bins import Bin, DiscretizedPositiveIntBin, WaneBin
from .strains import Strain


class Dimension(BaseModel):
    """A dimension of an compartment."""

    name: DynodeName = Field(
        description="""Dimension name, must be unique within a Compartment"""
    )
    bins: List[Bin] = Field(
        description="""Bins/cells within this dimension."""
    )

    def __len__(self):
        """Get len of a Dimension."""
        return len(self.bins)

    @property
    def idx(self):
        """Dimension idxs for indexing the bins within this dimension."""
        bin_namespace = SimpleNamespace()
        for bin_idx, single_bin in enumerate(self.bins):
            # build up the bins namespace for this compartment
            setattr(bin_namespace, single_bin.name, bin_idx)
        return bin_namespace

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
    def _validate_discretized_int_bins_sorted(
        cls, bins: list[Bin]
    ) -> list[Bin]:
        """Assert that DiscretizedPositiveIntBin do not overlap and sorts them lowest to highest."""
        assert len(bins) > 0, "can not have dimension with no bins"
        if all(isinstance(bin, DiscretizedPositiveIntBin) for bin in bins):
            # sort age bins with lowest min_value first
            bins_sorted = sorted(
                bins, key=lambda b: b.min_value, reverse=False
            )
            assert (
                bins == bins_sorted
            ), f"Any dimension made up of DiscretizedIntBins must be sorted, got {bins}"
            # assert that bins dont overlap now they are sorted
            assert all(
                [
                    bins[i].max_value < bins[i + 1].min_value
                    for i in range(len(bins) - 1)
                ]
            ), "DiscretizedPositiveIntBin within a dimension can not overlap."
        return bins

    @field_validator("bins", mode="after")
    @classmethod
    def _validate_no_gaps_discretized_int_bins(
        cls, bins: list[Bin]
    ) -> list[Bin]:
        """Validate that dimensions of DiscretizedPositiveIntBin have no gaps."""
        assert len(bins) > 0, "can not have dimension with no bins"
        if all(isinstance(bin, DiscretizedPositiveIntBin) for bin in bins):
            for i in range(len(bins) - 1):
                assert bins[i].max_value + 1 == bins[i + 1].min_value, (
                    f"dimensions containing DiscretizedPositiveIntBin can not "
                    f"have gaps between them, found one between "
                    f"{bins[i]} and {bins[i + 1]}"
                )

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
            DiscretizedPositiveIntBin(
                name=f"v{vax_count}", min_value=vax_count, max_value=vax_count
            )
            for vax_count in range(max_ordinal_vaccinations + 1)
        ]
        super().__init__(name="vax", bins=bins)

    @property
    def max_shots(self) -> int:
        """Maximum number of tracked vaccinations in the dimension.

        Additional shots do not increase the count.
        """
        # subtract 1 because we have a bin for 0 shots.
        return len(self.bins) - 1


class ImmuneHistoryDimension(Dimension):
    """A dimension meant to track how a population's immunity changes after recovering from a disease."""

    pass


class FullStratifiedImmuneHistoryDimension(ImmuneHistoryDimension):
    """A type of immune history which represents all possible unique infections."""

    def __init__(self, strains: list[Strain]) -> None:
        """Create a fully stratified immune history dimension."""
        strain_names = [s.strain_name for s in strains]
        all_immune_histories = [Bin(name="none")]
        for strain in range(1, len(strain_names) + 1):
            combs = combinations(strain_names, strain)
            all_immune_histories.extend(
                [Bin(name="_".join(comb)) for comb in combs]
            )

        super().__init__(name="hist", bins=all_immune_histories)


class LastStrainImmuneHistoryDimension(ImmuneHistoryDimension):
    """Immune history dimension that only tracks most recent infection."""

    def __init__(self, strains: list[Strain]) -> None:
        """Create an immune history dimension that only tracks last infected strain."""
        strain_names = [s.strain_name for s in strains]
        bins: list[Bin] = [Bin(name=state) for state in strain_names]
        bins.insert(0, Bin(name="none"))
        super().__init__(name="hist", bins=bins)


class WaneDimension(Dimension):
    """Dimension to tracking waning after recovery from a disease."""

    def __init__(
        self,
        waiting_times: list[PositiveFloat],
        base_protections: list[UnitIntervalFloat],
        name="wane",
    ):
        """Create a Dimension to track waning status.

        Parameters
        ----------
        waiting_times : list[PositiveFloat]
            A list of the waiting times of each bin from first wane bin to last.
        base_protections : list[UnitIntervalFloat]
            A list of base protections on [0, 1] for each waning bin, parallel to wait_times.
        name : str, optional
            name of the dimension, dimensions tracking different waning states
            must have different names, by default "wane".
        """
        assert (
            len(waiting_times) > 0
        ), "Wane dimension must have at least one bin."
        assert len(waiting_times) == len(
            base_protections
        ), "must pass equal length wait times and base protections"
        bins: list[Bin] = []
        for idx, (wait_time, base_protection) in enumerate(
            zip(waiting_times, base_protections)
        ):
            bins.append(
                WaneBin(
                    name=f"W{idx}",
                    waiting_time=wait_time,
                    base_protection=base_protection,
                )
            )
        super().__init__(
            name=name,
            bins=bins,
        )

    @model_validator(mode="after")
    def _validate_wane_bins_end_in_inf(self):
        """Validate last wane bin can not be waned out of."""
        last_wane_bin = self.bins[-1]
        assert isinstance(last_wane_bin, WaneBin)
        assert isinf(
            last_wane_bin.waiting_time
        ), "last wane bin should have math.inf waiting time"
        return self
