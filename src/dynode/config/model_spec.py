from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from . import (
    DataSpec,
    ParameterSpec,
    SimulationSpec,
)
from .bins import AgeBin
from .dimension import (
    FullStratifiedImmuneHistoryDimension,
    ImmuneHistoryDimension,
    LastStrainImmuneHistoryDimension,
)


class ModelSpec(BaseModel):
    """
    Top-level declarative specification for a dynamic ODE model.

    This class should validate cross-links among simulation structure,
    parameters, transmission/strains, initializer, and data.

    It should not:
    - sample NumPyro parameters
    - solve ODEs
    - mutate strain/config objects
    - construct JAX arrays directly
    """

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
        validate_assignment=True,
    )

    name: str = Field(
        description="Unique model name.",
    )

    version: str | None = Field(
        default=None,
        description="Optional model/spec version.",
    )

    description: str | None = Field(
        default=None,
        description="Optional human-readable model description.",
    )

    metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Optional user-defined metadata for auditability and run tracking.",
    )

    simulation: SimulationSpec = Field(
        description="Simulation structure: compartments, dimensions, initializer.",
    )

    parameters: ParameterSpec = Field(
        description="Parameter specification: priors, deterministic params, solver, transmission.",
    )

    data: DataSpec | None = Field(
        default=None,
        description="Optional observed-data specification.",
    )

    @property
    def transmission(self):
        return self.parameters.transmission

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

    @model_validator(mode="after")
    def validate_cross_links(self) -> Self:
        self._validate_immune_history_dimensions_match_strains()
        self._validate_introduced_strain_ages()
        self._validate_parameter_references()
        self._validate_initializer_against_model()
        self._validate_data_against_model()

        return self

    def _validate_immune_history_dimensions_match_strains(self) -> None:
        """
        Validate that immune-history dimensions were generated from the same
        strains defined in parameters.transmission.

        This replaces the old SimulationConfig._validate_immune_histories.
        """
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
                    f"Unsupported immune-history dimension type: "
                    f"{type(dim).__name__}."
                )

            custom_validator = getattr(dim, "validate_against_strains", None)

            if callable(custom_validator):
                custom_validator(self.strains)
                continue

            try:
                expected_dim = type(dim)(self.strains)
            except Exception as exc:
                raise TypeError(
                    f"Could not reconstruct immune-history dimension "
                    f"{dim.name!r} from ModelSpec strains. "
                    "Update the dimension constructor to accept StrainSpec "
                    "objects, or implement dimension.validate_against_strains(strains)."
                ) from exc

            if expected_dim != dim:
                raise ValueError(
                    f"Immune-history dimension {dim.name!r} does not match "
                    "parameters.transmission.strains. "
                    f"Expected {expected_dim!r}, found {dim!r}."
                )

    def _validate_introduced_strain_ages(self) -> None:
        """
        Validate that introduced strains only refer to age bins that exist in
        the simulation.

        This replaces the old SimulationConfig._validate_introduced_strains.
        """
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
                age
                for age in strain.introduction_ages
                if age not in age_bins
            ]

            if missing:
                raise ValueError(
                    f"Strain {strain.name!r} defines introduction_ages that "
                    f"are not present in the simulation age bins. "
                    f"Missing bins: {missing}."
                )

    def _validate_parameter_references(self) -> None:
        """
        Validate that ParamRef / DeterministicRef-style values point to known
        parameters.

        This is intentionally generic so the class can work while ParameterSpec
        is still evolving.
        """
        sampled_names = self._sampled_parameter_names()
        deterministic_names = self._deterministic_parameter_names()
        resolved_names = sampled_names | deterministic_names

        references = list(
            self._walk_references(
                self.parameters,
                path="parameters",
            )
        )

        if not references:
            return

        if not resolved_names:
            raise ValueError(
                "ModelSpec contains parameter references, but no parameter "
                "names could be inferred from ParameterSpec. Add properties "
                "such as sampled_parameter_names, deterministic_parameter_names, "
                "or resolved_parameter_names to ParameterSpec."
            )

        unknown_parameter_refs: list[str] = []
        unknown_deterministic_refs: list[str] = []

        for path, kind, name in references:
            if kind == "parameter" and name not in resolved_names:
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
        """
        Let InitializerSpec validate itself against the full model if it needs
        access to parameters, strains, or data.

        Simulation-only initializer validation can stay in SimulationSpec.
        """
        initializer = self.simulation.initializer

        validate_against_model = getattr(
            initializer,
            "validate_against_model",
            None,
        )

        if callable(validate_against_model):
            validate_against_model(self)

    def _validate_data_against_model(self) -> None:
        """
        Let DataSpec validate itself against the full model.

        Because DataSpec can vary widely across projects, this supports both:
        - explicit DataSpec hooks
        - a small number of common naming conventions
        """
        if self.data is None:
            return

        validate_against_model = getattr(
            self.data,
            "validate_against_model",
            None,
        )

        if callable(validate_against_model):
            validate_against_model(self)
            return

        validate_against_simulation = getattr(
            self.data,
            "validate_against_simulation",
            None,
        )

        if callable(validate_against_simulation):
            validate_against_simulation(self.simulation)
            return

        observed_compartments = self._get_data_observed_compartment_names()

        if observed_compartments is None:
            return

        missing = sorted(set(observed_compartments) - set(self.compartment_names))

        if missing:
            raise ValueError(
                "DataSpec refers to observed compartments that do not exist "
                f"in simulation.compartments: {missing}. "
                f"Known compartments are: {self.compartment_names}."
            )

    def introduction_age_masks(self) -> dict[str, list[int]]:
        """
        Build introduction-age masks without mutating StrainSpec.

        This replaces the old mutable introduction_ages_mask_vector behavior.
        """
        age_bins = self.age_bins

        return {
            strain.name: strain.introduction_age_mask(age_bins)
            for strain in self.strains
        }

    def interaction_matrix_spec(self):
        """
        Convenience proxy to TransmissionSpec.interaction_matrix_spec().
        """
        return self.transmission.interaction_matrix_spec()

    def interaction_matrix_named(self):
        """
        Convenience proxy to TransmissionSpec.interaction_matrix_named().
        """
        return self.transmission.interaction_matrix_named()

    def get_strain(self, name: str):
        return self.transmission.get_strain(name)

    def get_compartment(self, name: str):
        return self.simulation.get_compartment(name)

    def _sampled_parameter_names(self) -> set[str]:
        """
        Best-effort extraction of sampled parameter names from ParameterSpec.

        Ideally, ParameterSpec should eventually expose this directly.
        """
        direct = getattr(self.parameters, "sampled_parameter_names", None)

        if callable(direct):
            return set(direct())

        if direct is not None:
            return set(direct)

        resolved = getattr(self.parameters, "resolved_parameter_names", None)

        if callable(resolved):
            return set(resolved())

        if resolved is not None:
            return set(resolved)

        return self._names_from_parameter_container(
            getattr(self.parameters, "priors", None)
        )

    def _deterministic_parameter_names(self) -> set[str]:
        """
        Best-effort extraction of deterministic parameter names from ParameterSpec.

        Ideally, ParameterSpec should eventually expose this directly.
        """
        direct = getattr(self.parameters, "deterministic_parameter_names", None)

        if callable(direct):
            return set(direct())

        if direct is not None:
            return set(direct)

        names: set[str] = set()

        for attr_name in (
            "deterministic",
            "deterministics",
            "deterministic_params",
        ):
            names |= self._names_from_parameter_container(
                getattr(self.parameters, attr_name, None)
            )

        return names

    def _names_from_parameter_container(self, value: Any) -> set[str]:
        """
        Extract parameter names from common container shapes.

        Supports:
        - list[PriorSpec]
        - tuple[PriorSpec, ...]
        - dict[str, Any]
        - Pydantic models with named fields
        """
        if value is None:
            return set()

        if isinstance(value, dict):
            return set(value.keys())

        if isinstance(value, BaseModel):
            names = set(value.__class__.model_fields.keys())

            extra = getattr(value, "__pydantic_extra__", None)
            if extra:
                names |= set(extra.keys())

            return names

        if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
            names: set[str] = set()

            for item in value:
                name = getattr(item, "name", None)

                if name is not None:
                    names.add(name)

            return names

        return set()

    def _walk_references(
        self,
        value: Any,
        path: str,
    ) -> Iterable[tuple[str, str, str]]:
        """
        Recursively find ParamRef / DeterministicRef-style objects.

        This avoids importing concrete reference classes, which helps while
        the refactor is still in progress.

        Recognized by class names:
        - ParamRef
        - ParameterRef
        - DeterministicRef
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

    def _reference_kind(self, value: Any) -> str | None:
        cls_name = value.__class__.__name__

        if cls_name in {"ParamRef", "ParameterRef"} and hasattr(value, "name"):
            return "parameter"

        if cls_name == "DeterministicRef" and hasattr(value, "name"):
            return "deterministic"

        return None

    def _get_data_observed_compartment_names(self) -> list[str] | None:
        """
        Best-effort support for common DataSpec APIs.

        Recommended long-term approach:
        implement DataSpec.validate_against_model(model_spec).
        """
        if self.data is None:
            return None

        for attr_name in (
            "observed_compartment_names",
            "compartment_names",
            "required_compartments",
        ):
            value = getattr(self.data, attr_name, None)

            if callable(value):
                value = value()

            if value is not None:
                return list(value)

        observations = getattr(self.data, "observations", None)

        if observations is None:
            return None

        names: list[str] = []

        for observation in observations:
            for attr_name in (
                "compartment",
                "compartment_name",
                "source_compartment",
            ):
                value = getattr(observation, attr_name, None)

                if value is not None:
                    names.append(value)
                    break

        return names or None 
