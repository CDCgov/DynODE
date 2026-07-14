from __future__ import annotations

from math import isinf
from typing import Any, Literal

from pydantic import Field, PositiveFloat, model_validator
from typing_extensions import Self

from dynode.structure.bins.wane import WaneBin
from dynode.typing import DynodeName, UnitIntervalFloat

from .base import DimensionSpec


class WaneDimensionSpec(DimensionSpec):
    """
    Dimension for waning immunity after recovery or vaccination.
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
