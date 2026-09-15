from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from dynode.data.data_spec import DataSpec
from dynode.parameters.parameter_spec import ParameterBlockSpec
from dynode.solver.solver_spec import SolverSpec
from dynode.structure.bins.age import AgeBin
from dynode.structure.simulation.simulation import SimulationSpec
from dynode.transmission.model.spec import TransmissionSpec

from .references import reference_kind, walk_references
from .validation import (
    validate_data_against_model,
    validate_immune_history_dimensions_match_strains,
    validate_initializer_against_model,
    validate_introduced_strain_ages,
    validate_model_references,
)


class ModelSpec(BaseModel):
    """
    Top-level declarative specification for one ODE model instance.
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
        validate_immune_history_dimensions_match_strains(self)
        validate_introduced_strain_ages(self)
        validate_model_references(self)
        validate_initializer_against_model(self)
        validate_data_against_model(self)
        return self

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
    def _walk_references(cls, obj: Any, *, path: str):
        yield from walk_references(obj, path=path)

    @staticmethod
    def _reference_kind(obj: Any) -> str | None:
        return reference_kind(obj)
