from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import jax.numpy as jnp
import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    field_validator,
    model_validator,
)
from typing_extensions import Self

from dynode.typing import DynodeName


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


class DataFieldSpec(BaseModel):
    """Specification for an observed array, covariate, index array, or metadata field."""

    model_config = ConfigDict(
        extra="forbid", frozen=True, arbitrary_types_allowed=True
    )

    name: DynodeName
    kind: Literal["observed", "covariate", "index", "metadata"] = "observed"
    dims: tuple[str, ...] = Field(default_factory=tuple)
    dtype: Literal["float", "int", "bool", "date", "string", "object"] = (
        "float"
    )
    shape: tuple[int | None, ...] | None = None
    required: bool = True
    allow_missing: bool = False
    description: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)

    def validate_value(self, value: Any) -> None:
        if value is None:
            if self.required and not self.allow_missing:
                raise ValueError(
                    f"Data field {self.name!r} is required but value is None."
                )
            return
        if self.kind == "metadata" or self.dtype in {
            "date",
            "string",
            "object",
        }:
            return
        arr = np.asarray(value)
        if self.shape is not None:
            if len(arr.shape) != len(self.shape):
                raise ValueError(
                    f"Data field {self.name!r} expected ndim {len(self.shape)}, got shape {arr.shape}."
                )
            for idx, (actual, expected) in enumerate(
                zip(arr.shape, self.shape)
            ):
                if expected is not None and actual != expected:
                    raise ValueError(
                        f"Data field {self.name!r} axis {idx} expected size {expected}, got {actual}."
                    )
        if self.dtype == "float" and not np.issubdtype(arr.dtype, np.number):
            raise ValueError(f"Data field {self.name!r} must be numeric.")
        if self.dtype == "int" and not np.issubdtype(arr.dtype, np.integer):
            raise ValueError(
                f"Data field {self.name!r} must be integer typed."
            )


class DataSpec(BaseModel):
    """
    Declarative data contract.

    The older observed-series interface is preserved via `observations`. New
    experiments should prefer `fields`, which supports multidimensional arrays,
    covariates, index arrays, and metadata.
    """

    model_config = ConfigDict(
        extra="forbid", frozen=True, arbitrary_types_allowed=True
    )

    time: TimeSeriesSpec | None = None
    observations: tuple[ObservedSeriesSpec, ...] = Field(default_factory=tuple)
    fields: tuple[DataFieldSpec, ...] = Field(default_factory=tuple)
    metadata: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_data_spec(self) -> Self:
        names = self.observation_names + self.field_names
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(
                f"Data names must be unique. Duplicates: {duplicates}."
            )
        if self.time is not None:
            for obs in self.observations:
                if len(obs.values) != self.time.n_timepoints:
                    raise ValueError(
                        f"Observation {obs.name!r} length {len(obs.values)} does not match time length {self.time.n_timepoints}."
                    )
        return self

    @property
    def observation_names(self) -> list[str]:
        return [obs.name for obs in self.observations]

    @property
    def field_names(self) -> list[str]:
        return [field.name for field in self.fields]

    @property
    def data_names(self) -> list[str]:
        return self.observation_names + self.field_names

    def get_observation(self, name: str) -> ObservedSeriesSpec:
        for obs in self.observations:
            if obs.name == name:
                return obs
        raise KeyError(
            f"Unknown observation {name!r}. Known observations are: {self.observation_names}."
        )

    def get_field(self, name: str) -> DataFieldSpec:
        for field in self.fields:
            if field.name == name:
                return field
        raise KeyError(
            f"Unknown data field {name!r}. Known fields are: {self.field_names}."
        )

    def validate_values(self, values: Mapping[str, Any]) -> None:
        errors: list[str] = []
        for field in self.fields:
            if field.name not in values:
                if field.required:
                    errors.append(
                        f"Missing required data field {field.name!r}."
                    )
                continue
            try:
                field.validate_value(values[field.name])
            except ValueError as exc:
                errors.append(str(exc))
        if errors:
            raise ValueError("; ".join(errors))

    def validate_against_simulation(self, simulation: Any) -> None:
        compartment_names = set(getattr(simulation, "compartment_names", []))
        unknown = sorted(
            {obs.compartment_name for obs in self.observations}
            - compartment_names
        )
        if unknown:
            raise ValueError(
                f"Observed series refer to unknown compartments: {unknown}."
            )

    def validate_against_model(self, model: Any) -> None:
        self.validate_against_simulation(model.simulation)

    def as_jax_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if self.time is not None:
            result[self.time.name] = self.time.as_jax()
        for obs in self.observations:
            result[obs.name] = obs.as_jax()
        return result
