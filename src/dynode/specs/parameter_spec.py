from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from .deterministic_spec import DeterministicSpec
from .prior_spec import PriorSpec

ReferenceKind = Literal["parameter", "deterministic"]
ReferenceRecord = tuple[str, ReferenceKind, str]


class ParameterBlockSpec(BaseModel):
    """
    Reusable block of sampled and deterministic parameters.

    A ParameterBlockSpec is intentionally independent of solver and
    transmission configuration. It can be used for:
    - single-model local parameters
    - shared experiment-level parameters
    - instance/year-specific parameters in hierarchical experiments
    - observation-model parameters

    The older refactor put solver and transmission on ParameterSpec. That made
    a single model concise, but prevented shared parameter blocks from being
    first-class.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )

    name: str = Field(default="parameters")

    priors: tuple[PriorSpec, ...] = Field(default_factory=tuple)

    deterministic: tuple[DeterministicSpec, ...] = Field(default_factory=tuple)

    external_dependencies: tuple[str, ...] = Field(
        default_factory=tuple,
        description=(
            "Parameter names expected to be supplied by an outer context, "
            "for example shared parameters in an ExperimentSpec."
        ),
    )

    metadata: dict[str, str] = Field(default_factory=dict)

    @property
    def prior_names(self) -> list[str]:
        return [prior.name for prior in self.priors]

    @property
    def deterministic_names(self) -> list[str]:
        return [param.name for param in self.deterministic]

    @property
    def sampled_parameter_names(self) -> set[str]:
        return set(self.prior_names)

    @property
    def deterministic_parameter_names(self) -> set[str]:
        return set(self.deterministic_names)

    @property
    def local_parameter_names(self) -> set[str]:
        return (
            self.sampled_parameter_names | self.deterministic_parameter_names
        )

    @property
    def resolved_parameter_names(self) -> set[str]:
        return self.local_parameter_names | set(self.external_dependencies)

    @property
    def prior_map(self) -> dict[str, PriorSpec]:
        return {prior.name: prior for prior in self.priors}

    @property
    def deterministic_map(self) -> dict[str, DeterministicSpec]:
        return {param.name: param for param in self.deterministic}

    @model_validator(mode="after")
    def validate_parameter_block(self) -> Self:
        self._validate_unique_prior_names()
        self._validate_unique_deterministic_names()
        self._validate_no_prior_deterministic_name_collisions()
        self._validate_parameter_references()
        self._validate_deterministic_dependency_graph()
        return self

    def get_prior(self, name: str) -> PriorSpec:
        try:
            return self.prior_map[name]
        except KeyError as exc:
            raise KeyError(
                f"Unknown prior {name!r}. Known priors are: {self.prior_names}."
            ) from exc

    def get_deterministic(self, name: str) -> DeterministicSpec:
        try:
            return self.deterministic_map[name]
        except KeyError as exc:
            raise KeyError(
                f"Unknown deterministic parameter {name!r}. "
                f"Known deterministic parameters are: {self.deterministic_names}."
            ) from exc

    def has_parameter(self, name: str) -> bool:
        return name in self.resolved_parameter_names

    def deterministic_execution_order(self) -> list[DeterministicSpec]:
        deterministic_by_name = self.deterministic_map
        remaining = dict(deterministic_by_name)
        resolved = set(self.sampled_parameter_names) | set(
            self.external_dependencies
        )
        ordered: list[DeterministicSpec] = []

        while remaining:
            ready_names = [
                name
                for name, spec in remaining.items()
                if self._dependencies_of_deterministic(spec) <= resolved
            ]

            if not ready_names:
                unresolved = {
                    name: sorted(
                        self._dependencies_of_deterministic(spec) - resolved
                    )
                    for name, spec in remaining.items()
                }
                raise ValueError(
                    "Could not determine deterministic parameter execution "
                    "order. This usually indicates a cycle or unresolved "
                    f"dependency. Remaining dependencies: {unresolved}."
                )

            for name in ready_names:
                spec = remaining.pop(name)
                ordered.append(spec)
                resolved.add(name)

        return ordered

    @staticmethod
    def _duplicates(values: Iterable[str]) -> list[str]:
        values = list(values)
        return sorted({value for value in values if values.count(value) > 1})

    def _validate_unique_prior_names(self) -> None:
        duplicates = self._duplicates(self.prior_names)
        if duplicates:
            raise ValueError(
                f"Prior names must be unique. Duplicates: {duplicates}."
            )

    def _validate_unique_deterministic_names(self) -> None:
        duplicates = self._duplicates(self.deterministic_names)
        if duplicates:
            raise ValueError(
                "Deterministic parameter names must be unique. "
                f"Duplicates: {duplicates}."
            )

    def _validate_no_prior_deterministic_name_collisions(self) -> None:
        collisions = sorted(
            self.sampled_parameter_names & self.deterministic_parameter_names
        )
        if collisions:
            raise ValueError(
                "A parameter cannot be both sampled and deterministic. "
                f"Colliding names: {collisions}."
            )

    def _validate_parameter_references(self) -> None:
        references = list(self._walk_references(self, path="parameters"))
        unknown_parameter_refs: list[str] = []
        unknown_deterministic_refs: list[str] = []

        for path, kind, name in references:
            if (
                kind == "parameter"
                and name not in self.resolved_parameter_names
            ):
                unknown_parameter_refs.append(f"{path} -> {name!r}")
            if (
                kind == "deterministic"
                and name not in self.deterministic_parameter_names
            ):
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

    def _validate_deterministic_dependency_graph(self) -> None:
        self.deterministic_execution_order()

    @staticmethod
    def _dependencies_of_deterministic(spec: DeterministicSpec) -> set[str]:
        deps = getattr(spec, "dependencies", set())
        if callable(deps):
            deps = deps()
        return set(deps or set())

    @classmethod
    def _walk_references(
        cls,
        obj: Any,
        *,
        path: str,
    ) -> Iterable[ReferenceRecord]:
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
                value = getattr(obj, field_name)
                yield from cls._walk_references(
                    value, path=f"{path}.{field_name}"
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
    def _reference_kind(obj: Any) -> ReferenceKind | None:
        class_name = type(obj).__name__
        if class_name in {"ParamRef", "ParameterRef"}:
            return "parameter"
        if class_name == "DeterministicRef":
            return "deterministic"
        return None


class ModelParameterSpec(ParameterBlockSpec):
    """Local parameter block for a single ModelSpec."""


class ParameterSpec(ParameterBlockSpec):
    """
    Backward-compatible name for a parameter block.

    The new architecture uses ParameterBlockSpec for both local and shared
    parameter groups. SolverSpec and TransmissionSpec now belong on ModelSpec,
    not on ParameterSpec. Older dict configs with parameters.solver or
    parameters.transmission are migrated by ModelSpec's before-validator.
    """
