from __future__ import annotations

from typing import Any, Literal

import jax.numpy as jnp
import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    PositiveFloat,
    field_validator,
    model_validator,
)
from typing_extensions import Self

from dynode.typing import DynodeName


class TimeSeriesSpec(BaseModel):
    """
    Shared time axis for observed data.

    Values are usually model time, such as days since simulation start.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    values: tuple[NonNegativeFloat, ...] = Field(
        min_length=1,
        description="Observation times, usually days since model start.",
    )

    unit: str = Field(
        default="days",
        description="Unit for the time axis.",
    )

    name: str = Field(
        default="time",
        description="Name of the time axis.",
    )

    @field_validator("values")
    @classmethod
    def validate_time_is_strictly_increasing(
        cls,
        values: tuple[float, ...],
    ) -> tuple[float, ...]:
        if any(next_value <= current for current, next_value in zip(values, values[1:])):
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
    """
    One observed time series.

    Example
    -------
    Weekly observed cases from the infectious compartment:

        name="cases"
        compartment_name="I"
        values=(10, 14, 18, 12)
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )

    name: DynodeName = Field(
        description="Unique name for this observed series.",
    )

    compartment_name: DynodeName = Field(
        description=(
            "Name of the model compartment this observation is derived from. "
            "This should match one of SimulationSpec.compartments."
        ),
    )

    values: tuple[float | None, ...] = Field(
        min_length=1,
        description=(
            "Observed values. Use None for missing observations. "
            "The length must match DataSpec.time."
        ),
    )

    scale: Literal[
        "count",
        "rate",
        "proportion",
        "continuous",
        "log",
    ] = Field(
        default="count",
        description="Semantic scale of the observed values.",
    )

    unit: str | None = Field(
        default=None,
        description="Optional unit label, such as cases, admissions, deaths.",
    )

    lower_bound: float | None = Field(
        default=None,
        description="Optional lower bound for observed values.",
    )

    upper_bound: float | None = Field(
        default=None,
        description="Optional upper bound for observed values.",
    )

    metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Optional metadata for this observed series.",
    )

    @model_validator(mode="after")
    def validate_values(self) -> Self:
        numeric_values = [
            value
            for value in self.values
            if value is not None
        ]

        if self.scale == "count":
            bad_values = [
                value
                for value in numeric_values
                if value < 0 or int(value) != value
            ]

            if bad_values:
                raise ValueError(
                    f"Count observations must be non-negative integer-like values. "
                    f"Bad values in {self.name!r}: {bad_values}."
                )

        if self.scale in {"rate", "proportion"}:
            bad_values = [
                value
                for value in numeric_values
                if value < 0
            ]

            if bad_values:
                raise ValueError(
                    f"{self.scale!r} observations must be non-negative. "
                    f"Bad values in {self.name!r}: {bad_values}."
                )

        if self.scale == "proportion":
            bad_values = [
                value
                for value in numeric_values
                if value > 1
            ]

            if bad_values:
                raise ValueError(
                    f"Proportion observations must be <= 1. "
                    f"Bad values in {self.name!r}: {bad_values}."
                )

        if self.lower_bound is not None:
            below = [
                value
                for value in numeric_values
                if value < self.lower_bound
            ]

            if below:
                raise ValueError(
                    f"Values in {self.name!r} fall below lower_bound="
                    f"{self.lower_bound}: {below}."
                )

        if self.upper_bound is not None:
            above = [
                value
                for value in numeric_values
                if value > self.upper_bound
            ]

            if above:
                raise ValueError(
                    f"Values in {self.name!r} exceed upper_bound="
                    f"{self.upper_bound}: {above}."
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
        """
        True where the observation is missing.
        """
        return tuple(value is None for value in self.values)

    @property
    def observed_mask(self) -> tuple[bool, ...]:
        """
        True where the observation is present.
        """
        return tuple(value is not None for value in self.values)

    def values_with_nan(self) -> tuple[float, ...]:
        """
        Convert None to NaN for array conversion.

        NumPyro likelihood code may instead prefer explicit masking.
        """
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


class DataSpec(BaseModel):
    """
    Declarative observed-data specification.

    Responsibilities:
    - validate observation time axis
    - validate observed series lengths
    - validate unique observation names
    - validate basic value constraints
    - expose runtime-friendly array helpers
    - validate data references against SimulationSpec or ModelSpec

    This class should not:
    - define NumPyro likelihoods
    - call numpyro.sample
    - solve ODEs
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )

    time: TimeSeriesSpec = Field(
        description="Shared time axis for all observations.",
    )

    observations: tuple[ObservedSeriesSpec, ...] = Field(
        default_factory=tuple,
        description="Observed time series.",
    )

    metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Optional dataset-level metadata.",
    )

    @property
    def n_timepoints(self) -> int:
        return self.time.n_timepoints

    @property
    def observation_names(self) -> list[str]:
        return [observation.name for observation in self.observations]

    @property
    def observed_compartment_names(self) -> list[str]:
        return [
            observation.compartment_name
            for observation in self.observations
        ]

    @property
    def unique_observed_compartment_names(self) -> list[str]:
        seen: set[str] = set()
        names: list[str] = []

        for name in self.observed_compartment_names:
            if name not in seen:
                seen.add(name)
                names.append(name)

        return names

    @model_validator(mode="after")
    def validate_data_spec(self) -> Self:
        self._validate_unique_observation_names()
        self._validate_observation_lengths()

        return self

    def _validate_unique_observation_names(self) -> None:
        names = self.observation_names
        duplicates = sorted({name for name in names if names.count(name) > 1})

        if duplicates:
            raise ValueError(
                f"Observation names must be unique. Duplicates: {duplicates}."
            )

    def _validate_observation_lengths(self) -> None:
        bad_lengths = {
            observation.name: len(observation.values)
            for observation in self.observations
            if len(observation.values) != self.n_timepoints
        }

        if bad_lengths:
            raise ValueError(
                "All observed series must have the same length as time.values. "
                f"Expected {self.n_timepoints}, got {bad_lengths}."
            )

    def get_observation(self, name: str) -> ObservedSeriesSpec:
        for observation in self.observations:
            if observation.name == name:
                return observation

        raise KeyError(
            f"Unknown observation {name!r}. "
            f"Known observations are: {self.observation_names}."
        )

    def observations_for_compartment(
        self,
        compartment_name: str,
    ) -> list[ObservedSeriesSpec]:
        return [
            observation
            for observation in self.observations
            if observation.compartment_name == compartment_name
        ]

    def validate_against_simulation(self, simulation) -> None:
        """
        Validate that observed compartments exist in SimulationSpec.
        """
        missing = sorted(
            set(self.observed_compartment_names)
            - set(simulation.compartment_names)
        )

        if missing:
            raise ValueError(
                "DataSpec refers to compartments that do not exist in "
                f"SimulationSpec: {missing}. "
                f"Known compartments are: {simulation.compartment_names}."
            )

    def validate_against_model(self, model) -> None:
        """
        Validate DataSpec against the full ModelSpec.

        Kept separate from validate_against_simulation so later this can also
        check parameters, observation models, reporting rates, etc.
        """
        self.validate_against_simulation(model.simulation)

    def as_jax(self) -> dict[str, Any]:
        """
        Runtime helper returning JAX arrays.

        Missing observations are represented as NaN, with a separate boolean
        mask indicating which observations are present.
        """
        return {
            "time": self.time.as_jax(),
            "observations": {
                observation.name: observation.as_jax()
                for observation in self.observations
            },
            "observed_masks": {
                observation.name: observation.mask_as_jax()
                for observation in self.observations
            },
            "compartment_names": {
                observation.name: observation.compartment_name
                for observation in self.observations
            },
        }

    def as_numpy(self) -> dict[str, Any]:
        """
        Runtime/helper equivalent using NumPy arrays.
        """
        return {
            "time": self.time.as_numpy(),
            "observations": {
                observation.name: observation.as_numpy()
                for observation in self.observations
            },
            "observed_masks": {
                observation.name: observation.mask_as_numpy()
                for observation in self.observations
            },
            "compartment_names": {
                observation.name: observation.compartment_name
                for observation in self.observations
            },
        }

    def model_data_for_observe_fn(self) -> dict[str, Any]:
        """
        Convenience method for passing validated data into observe_fn.

        Your observe_fn can accept this dictionary instead of the Pydantic model
        if you want to keep NumPyro/JAX execution away from Pydantic objects.
        """
        return self.as_jax()
