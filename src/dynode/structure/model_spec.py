from __future__ import annotations

from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from dynode.data.data_spec import DataSpec
from dynode.parameters.parameter_spec import ParameterBlockSpec
from dynode.solver.solver_spec import SolverSpec
from dynode.structure.bin_spec import AgeBin
from dynode.structure.dimension_spec import (
    FullStratifiedImmuneHistoryDimension,
    ImmuneHistoryDimension,
    LastStrainImmuneHistoryDimension,
)
from dynode.structure.simulation_spec import SimulationSpec
from dynode.transmission.transmission_spec import TransmissionSpec


class ModelSpec(BaseModel):
    """
    Top-level declarative specification for one ODE model instance.

    Solver and transmission are first-class model fields. ParameterSpec is now
    a reusable parameter block and no longer owns solver/transmission.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )

    name: str
    version: str | None = None
    description: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)

    simulation: SimulationSpec
    solver: SolverSpec
    transmission: TransmissionSpec
    parameters: ParameterBlockSpec = Field(default_factory=ParameterBlockSpec)
    data: DataSpec | None = None

    external_parameter_names: tuple[str, ...] = Field(
        default_factory=tuple,
        description=(
            "Parameter names supplied by an outer ExperimentSpec or caller. "
            "Used for validating strain/initializer references that are not "
            "local to this ModelSpec."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_parameter_shape(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        parameters = data.get("parameters")
        if isinstance(parameters, dict):
            parameters = dict(parameters)
            if "solver" not in data and "solver" in parameters:
                data["solver"] = parameters.pop("solver")
            if "transmission" not in data and "transmission" in parameters:
                data["transmission"] = parameters.pop("transmission")
            data["parameters"] = parameters
        return data

    @property
    def strains(self):
        return self.transmission.strains

    @property
    def strain_names(self) -> list[str]:
        return [strain.name for strain in self.strains]

    @property
    def compartment_names(self) -> list[str]:
        return self.simulation.compartment_names

    @property
    def age_bins(self) -> list[AgeBin]:
        return self.simulation.get_age_bins()

    @property
    def available_parameter_names(self) -> set[str]:
        return set(self.parameters.resolved_parameter_names) | set(
            self.external_parameter_names
        )

    @model_validator(mode="after")
    def validate_cross_links(self) -> Self:
        self._validate_immune_history_dimensions_match_strains()
        self._validate_introduced_strain_ages()
        self._validate_model_references()
        self._validate_initializer_against_model()
        self._validate_data_against_model()
        return self

    def _validate_immune_history_dimensions_match_strains(self) -> None:
        immune_dims = [
            dim
            for dim in self.simulation.flatten_dims()
            if isinstance(dim, ImmuneHistoryDimension)
        ]
        for dim in immune_dims:
            if not isinstance(
                dim,
                (
                    FullStratifiedImmuneHistoryDimension,
                    LastStrainImmuneHistoryDimension,
                ),
            ):
                raise ValueError(
                    f"Unsupported immune-history dimension type: {type(dim).__name__}."
                )
            dim.validate_against_strains(self.strains)

    def _validate_introduced_strain_ages(self) -> None:
        strains_with_intro_ages = [
            strain
            for strain in self.strains
            if strain.is_introduced and strain.introduction_ages is not None
        ]
        if not strains_with_intro_ages:
            return
        age_bins = self.age_bins
        if not age_bins:
            names = [strain.name for strain in strains_with_intro_ages]
            raise ValueError(
                "Some introduced strains define introduction_ages, but the "
                f"simulation has no age dimension. Strains: {names}."
            )
        for strain in strains_with_intro_ages:
            missing = [
                age for age in strain.introduction_ages if age not in age_bins
            ]
            if missing:
                raise ValueError(
                    f"Strain {strain.name!r} defines introduction_ages that are not "
                    f"present in the simulation age bins. Missing bins: {missing}."
                )

    def _validate_model_references(self) -> None:
        available = self.available_parameter_names
        deterministic_names = set(
            self.parameters.deterministic_parameter_names
        ) | set(self.external_parameter_names)
        references = list(
            self._walk_references(self.transmission, path="transmission")
        )
        references += list(
            self._walk_references(
                self.simulation.initializer, path="simulation.initializer"
            )
        )
        unknown_parameter_refs: list[str] = []
        unknown_deterministic_refs: list[str] = []
        for path, kind, name in references:
            if kind == "parameter" and name not in available:
                unknown_parameter_refs.append(f"{path} -> {name!r}")
            if kind == "deterministic" and name not in deterministic_names:
                unknown_deterministic_refs.append(f"{path} -> {name!r}")
        errors: list[str] = []
        if unknown_parameter_refs:
            errors.append(
                "Unknown parameter references: "
                + ", ".join(unknown_parameter_refs)
            )
        if unknown_deterministic_refs:
            errors.append(
                "Unknown deterministic references: "
                + ", ".join(unknown_deterministic_refs)
            )
        if errors:
            raise ValueError("; ".join(errors))

    def _validate_initializer_against_model(self) -> None:
        validate_against_model = getattr(
            self.simulation.initializer, "validate_against_model", None
        )
        if callable(validate_against_model):
            validate_against_model(self)

    def _validate_data_against_model(self) -> None:
        if self.data is None:
            return
        validate_against_model = getattr(
            self.data, "validate_against_model", None
        )
        if callable(validate_against_model):
            validate_against_model(self)

    def introduction_age_masks(self) -> dict[str, list[int]]:
        return {
            strain.name: strain.introduction_age_mask(self.age_bins)
            for strain in self.strains
            if strain.is_introduced
        }

    def interaction_matrix_spec(self):
        return self.transmission.interaction_matrix_spec()

    def interaction_matrix_named(self):
        return self.transmission.interaction_matrix_named()

    def get_strain(self, name: str):
        return self.transmission.get_strain(name)

    def get_compartment(self, name: str):
        return self.simulation.get_compartment(name)

    @classmethod
    def _walk_references(
        cls, obj: Any, *, path: str
    ) -> Iterable[tuple[str, str, str]]:
        if obj is None:
            return
        kind = cls._reference_kind(obj)
        if kind is not None:
            name = getattr(obj, "name", None)
            if name is not None:
                yield (path, kind, str(name))
            return
        if isinstance(obj, BaseModel):
            for field_name in obj.model_fields:
                yield from cls._walk_references(
                    getattr(obj, field_name), path=f"{path}.{field_name}"
                )
            return
        if isinstance(obj, dict):
            for key, value in obj.items():
                yield from cls._walk_references(value, path=f"{path}[{key!r}]")
            return
        if isinstance(obj, (list, tuple, set, frozenset)):
            for idx, value in enumerate(obj):
                yield from cls._walk_references(value, path=f"{path}[{idx}]")
            return

    @staticmethod
    def _reference_kind(obj: Any) -> str | None:
        class_name = type(obj).__name__
        if class_name in {"ParamRef", "ParameterRef"}:
            return "parameter"
        if class_name == "DeterministicRef":
            return "deterministic"
        return None
