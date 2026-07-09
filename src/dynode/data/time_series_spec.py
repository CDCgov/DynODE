from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    field_validator,
)


class TimeSeriesSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    values: tuple[NonNegativeFloat, ...] = Field(min_length=1)
    unit: str = "days"
    name: str = "time"

    @field_validator("values")
    @classmethod
    def validate_time_is_strictly_increasing(
        cls, values: tuple[float, ...]
    ) -> tuple[float, ...]:
        if any(
            next_value <= current
            for current, next_value in zip(values, values[1:])
        ):
            raise ValueError("Time values must be strictly increasing.")
        return values

    @property
    def n_timepoints(self) -> int:
        return len(self.values)

    def as_jax(self):
        return jnp.asarray(self.values)

    def as_numpy(self):
        return np.asarray(self.values)
