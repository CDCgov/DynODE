from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from .data_spec import DataSpec
from .model_spec import ModelSpec
from .parameter_spec import ParameterBlockSpec


class ModelInstanceSpec(BaseModel):
    """One named instance of a ModelSpec within a larger experiment."""

    model_config = ConfigDict(
        extra="forbid", frozen=True, arbitrary_types_allowed=True
    )

    key: str
    model: ModelSpec
    parameters: ParameterBlockSpec = Field(default_factory=ParameterBlockSpec)
    data: DataSpec | None = None
    static_context: dict[str, Any] = Field(default_factory=dict)
    t0: float | None = None
    t1: float | None = None
    metadata: dict[str, str] = Field(default_factory=dict)

    @property
    def local_parameter_names(self) -> set[str]:
        return self.parameters.local_parameter_names


class ExperimentSpec(BaseModel):
    """Experiment-level spec for multi-instance and hierarchical models."""

    model_config = ConfigDict(
        extra="forbid", frozen=True, arbitrary_types_allowed=True
    )

    name: str
    version: str | None = None
    description: str | None = None
    shared_parameters: ParameterBlockSpec = Field(
        default_factory=lambda: ParameterBlockSpec(name="shared")
    )
    instances: tuple[ModelInstanceSpec, ...] = Field(min_length=1)
    metadata: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_experiment(self) -> Self:
        keys = [instance.key for instance in self.instances]
        duplicates = sorted({key for key in keys if keys.count(key) > 1})
        if duplicates:
            raise ValueError(
                f"Experiment instance keys must be unique. Duplicates: {duplicates}."
            )
        shared_names = self.shared_parameters.resolved_parameter_names
        for instance in self.instances:
            available = _available_instance_names(
                shared_names=shared_names,
                instance=instance,
            )
            model_required = set(instance.model.external_parameter_names)
            missing_model = sorted(model_required - available)
            if missing_model:
                raise ValueError(
                    f"Instance {instance.key!r} model expects external parameters {missing_model}, "
                    f"but available experiment parameters are {sorted(available)}."
                )
            for prior in instance.parameters.priors:
                missing = sorted(prior.dependencies - available)
                if missing:
                    raise ValueError(
                        f"Instance {instance.key!r} prior {prior.name!r} depends on unknown parameters {missing}."
                    )
            for deterministic in instance.parameters.deterministic:
                missing = sorted(deterministic.dependencies - available)
                if missing:
                    raise ValueError(
                        f"Instance {instance.key!r} deterministic {deterministic.name!r} depends on unknown parameters {missing}."
                    )
        return self


def _available_instance_names(
    *,
    shared_names: set[str],
    instance: ModelInstanceSpec,
) -> set[str]:
    """Return names available to one experiment instance.

    Includes shared parameters, instance-local parameters, top-level static
    context keys, and keys nested under static_context["parameter_context"].
    The nested parameter_context pattern is useful for static per-instance
    values such as nu, ve_infection, and beta modifier arrays.
    """
    available = set(shared_names)
    available |= set(instance.parameters.resolved_parameter_names)
    available |= {str(name) for name in instance.static_context}

    parameter_context = instance.static_context.get("parameter_context")
    if isinstance(parameter_context, dict):
        available |= {str(name) for name in parameter_context}

    return available
