from __future__ import annotations

from typing import Literal

import jax.numpy as jnp
import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)
from typing_extensions import Self

from dynode.typing import DynodeName


class ObservedSeriesSpec(BaseModel):
    """Backward-compatible scalar time-series observation spec."""

    model_config = ConfigDict(
        extra="forbid", frozen=True, arbitrary_types_allowed=True
    )

    name: DynodeName
    compartment_name: DynodeName
    values: tuple[float | None, ...] = Field(min_length=1)
    scale: Literal["count", "rate", "proportion", "continuous", "log"] = (
        "count"
    )
    unit: str | None = None
    lower_bound: float | None = None
    upper_bound: float | None = None
    metadata: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_values(self) -> Self:
        numeric_values = [value for value in self.values if value is not None]
        if self.scale == "count":
            bad_values = [
                value
                for value in numeric_values
                if value < 0 or int(value) != value
            ]
            if bad_values:
                raise ValueError(
                    f"Count observations must be non-negative integer-like values. Bad values in {self.name!r}: {bad_values}."
                )
        if self.scale in {"rate", "proportion"}:
            bad_values = [value for value in numeric_values if value < 0]
            if bad_values:
                raise ValueError(
                    f"{self.scale!r} observations must be non-negative. Bad values in {self.name!r}: {bad_values}."
                )
        if self.scale == "proportion":
            bad_values = [value for value in numeric_values if value > 1]
            if bad_values:
                raise ValueError(
                    f"Proportion observations must be <= 1. Bad values in {self.name!r}: {bad_values}."
                )
        if self.lower_bound is not None:
            below = [
                value for value in numeric_values if value < self.lower_bound
            ]
            if below:
                raise ValueError(
                    f"Values in {self.name!r} fall below lower_bound={self.lower_bound}: {below}."
                )
        if self.upper_bound is not None:
            above = [
                value for value in numeric_values if value > self.upper_bound
            ]
            if above:
                raise ValueError(
                    f"Values in {self.name!r} exceed upper_bound={self.upper_bound}: {above}."
                )
        if (
            self.lower_bound is not None
            and self.upper_bound is not None
            and self.upper_bound < self.lower_bound
        ):
            raise ValueError(
                f"upper_bound must be >= lower_bound for {self.name!r}."
            )
        return self

    @property
    def missing_mask(self) -> tuple[bool, ...]:
        return tuple(value is None for value in self.values)

    @property
    def observed_mask(self) -> tuple[bool, ...]:
        return tuple(value is not None for value in self.values)

    def values_with_nan(self) -> tuple[float, ...]:
        return tuple(
            float("nan") if value is None else float(value)
            for value in self.values
        )

    def as_jax(self):
        return jnp.asarray(self.values_with_nan())

    def mask_as_jax(self):
        return jnp.asarray(self.observed_mask)

    def as_numpy(self):
        return np.asarray(self.values_with_nan())

    def mask_as_numpy(self):
        return np.asarray(self.observed_mask)
