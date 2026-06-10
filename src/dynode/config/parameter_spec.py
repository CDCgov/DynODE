from collections.abc import Iterable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from . import (
    DeterministicSpec,
    PriorSpec,
    SolverSpec,
    TransmissionSpec,
)

ReferenceKind = Literal["parameter", "deterministic"]
ReferenceRecord = tuple[str, ReferenceKind, str]


class ParameterSpec(BaseModel):
    """
    Declarative parameter specification for a dynamic ODE model.

    Responsibilities:
    - validate prior names
    - validate deterministic parameter names
    - validate references to parameters
    - expose useful parameter-name maps
    - determine deterministic evaluation order

    This class should not:
    - call numpyro.sample
    - call numpyro.deterministic
    - mutate parameters during validation
    - build JAX arrays
    """

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
        validate_assignment=True,
    )

    solver: SolverSpec = Field(
        description="ODE solver settings.",
    )

    transmission: TransmissionSpec = Field(
        description="Transmission and strain-level parameter specification.",
    )

    priors: tuple[PriorSpec, ...] = Field(
        default_factory=tuple,
        description="Sampled parameters and their prior distributions.",
    )

    deterministic: tuple[DeterministicSpec, ...] = Field(
        default_factory=tuple,
        description="Deterministic parameters derived from sampled or deterministic values.",
    )

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
    def resolved_parameter_names(self) -> set[str]:
        return (
            self.sampled_parameter_names | self.deterministic_parameter_names
        )

    @property
    def prior_map(self) -> dict[str, PriorSpec]:
        return {prior.name: prior for prior in self.priors}

    @property
    def deterministic_map(self) -> dict[str, DeterministicSpec]:
        return {param.name: param for param in self.deterministic}

    @model_validator(mode="after")
    def validate_parameter_spec(self) -> Self:
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
        """
        Return deterministic parameters in dependency-safe order.

        Example
        -------
        If:

            beta = sampled
            gamma = sampled
            r0 = beta / gamma
            log_r0 = log(r0)

        Then the deterministic order should be:

            r0, log_r0
        """
        deterministic_by_name = self.deterministic_map
        remaining = dict(deterministic_by_name)
        resolved = set(self.sampled_parameter_names)
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
                    "Could not determine deterministic parameter execution order. "
                    "This usually indicates a cycle or an unresolved dependency. "
                    f"Remaining dependencies: {unresolved}."
                )

            for name in ready_names:
                spec = remaining.pop(name)
                ordered.append(spec)
                resolved.add(name)

        return ordered

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
        """
        Validate ParamRef / DeterministicRef-style references.

        This checks references inside:
        - priors
        - deterministic specs
        - transmission specs
        - strain specs
        - interaction specs

        Recognition is intentionally generic while your refactor is still
        evolving. It recognizes classes named:
        - ParamRef
        - ParameterRef
        - DeterministicRef
        """
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
        """
        Validate deterministic dependencies.

        This catches:
        - references to missing parameters
        - deterministic cycles
        - deterministic parameters that cannot be ordered
        """
        self.deterministic_execution_order()

    def _dependencies_of_deterministic(
        self,
        spec: DeterministicSpec,
    ) -> set[str]:
        """
        Infer dependencies of a DeterministicSpec.

        Preferred long-term design:
        DeterministicSpec should expose one of:

            dependencies
            dependency_names
            depends_on

        This fallback also recursively walks references inside the spec.
        """
        for attr_name in (
            "dependencies",
            "dependency_names",
            "depends_on",
        ):
            value = getattr(spec, attr_name, None)

            if callable(value):
                value = value()

            if value is not None:
                return set(value)

        dependencies: set[str] = set()

        for _, kind, name in self._walk_references(spec, path=spec.name):
            if kind in {"parameter", "deterministic"}:
                dependencies.add(name)

        dependencies.discard(spec.name)

        return dependencies

    def _walk_references(
        self,
        value: Any,
        path: str,
    ) -> Iterable[ReferenceRecord]:
        """
        Recursively walk an object and yield parameter references.

        This makes ParameterSpec tolerant of your in-progress class design.
        """
        kind = self._reference_kind(value)

        if kind is not None:
            yield path, kind, value.name
            return

        if isinstance(value, BaseModel):
            for field_name in value.__class__.model_fields:
                field_value = getattr(value, field_name)

                yield from self._walk_references(
                    field_value,
                    path=f"{path}.{field_name}",
                )

            extra = getattr(value, "__pydantic_extra__", None)

            if extra:
                for key, extra_value in extra.items():
                    yield from self._walk_references(
                        extra_value,
                        path=f"{path}.{key}",
                    )

            return

        if isinstance(value, dict):
            for key, item in value.items():
                yield from self._walk_references(
                    item,
                    path=f"{path}[{key!r}]",
                )

            return

        if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
            for i, item in enumerate(value):
                yield from self._walk_references(
                    item,
                    path=f"{path}[{i}]",
                )

    def _reference_kind(self, value: Any) -> ReferenceKind | None:
        cls_name = value.__class__.__name__

        if cls_name in {"ParamRef", "ParameterRef"} and hasattr(value, "name"):
            return "parameter"

        if cls_name == "DeterministicRef" and hasattr(value, "name"):
            return "deterministic"

        return None

    @staticmethod
    def _duplicates(values: list[str]) -> list[str]:
        return sorted({value for value in values if values.count(value) > 1})
