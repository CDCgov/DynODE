from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)
from typing_extensions import Self

from .data_field_spec import DataFieldSpec
from .observed_series_spec import ObservedSeriesSpec
from .time_series_spec import TimeSeriesSpec


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
