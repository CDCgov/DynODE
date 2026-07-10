from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, NonNegativeInt, model_validator

from dynode.typing import DynodeName

from dynode.structure.bins.discretized import DiscretizedPositiveIntBin
from .base import DimensionSpec


class VaccinationDimensionSpec(DimensionSpec):
    """
    Vaccination dimension.

    Tracks ordinal vaccine dose count, optionally including one seasonal dose.
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
