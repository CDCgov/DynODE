from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from dynode.value.constant import ConstantValueSpec
from dynode.value.references import DeterministicRef, ParamRef
from dynode.value.unions import InteractionValue

from .coercion import coerce_interaction_data
from .validation import (
    validate_bounds_are_ordered,
    validate_constant_value_bounds,
    validate_no_data_dependencies,
)


class InteractionSpec(BaseModel):
    """
    Declarative strain-interaction specification.

    This represents one entry in the strain interaction matrix:

        interaction[source_strain, target_strain]
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )

    value: InteractionValue = Field(
        description=(
            "Interaction value. May be a constant, parameter reference, "
            "deterministic reference, or expression."
        )
    )

    lower_bound: float | None = Field(
        default=0.0,
        description=(
            "Optional lower bound for constant interaction values. "
            "Default is 0.0 because strain interactions are usually non-negative."
        ),
    )

    upper_bound: float | None = Field(
        default=None,
        description=(
            "Optional upper bound for constant interaction values. "
            "Set to 1.0 if the interaction represents a probability or proportion."
        ),
    )

    description: str | None = Field(
        default=None,
        description="Optional human-readable description.",
    )

    metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Optional metadata for auditing, documentation, or UI display.",
    )

    @model_validator(mode="before")
    @classmethod
    def coerce_interaction(cls, data: Any) -> Any:
        """Allow compact interaction forms."""
        if isinstance(data, cls):
            return data

        return coerce_interaction_data(data)

    @model_validator(mode="after")
    def validate_interaction(self) -> Self:
        validate_bounds_are_ordered(
            lower_bound=self.lower_bound,
            upper_bound=self.upper_bound,
        )
        validate_constant_value_bounds(
            value=self.value,
            lower_bound=self.lower_bound,
            upper_bound=self.upper_bound,
        )
        validate_no_data_dependencies(self.data_dependencies)
        return self

    @classmethod
    def fixed(
        cls,
        value: int | float | list[int] | list[float],
        *,
        lower_bound: float | None = 0.0,
        upper_bound: float | None = None,
        description: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> InteractionSpec:
        """Construct a fixed constant interaction."""
        return cls(
            value=ConstantValueSpec(value=value),
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            description=description,
            metadata=metadata or {},
        )

    @classmethod
    def parameter(
        cls,
        name: str,
        *,
        lower_bound: float | None = 0.0,
        upper_bound: float | None = None,
        description: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> InteractionSpec:
        """Construct an interaction from a sampled parameter reference."""
        return cls(
            value=ParamRef(name=name),
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            description=description,
            metadata=metadata or {},
        )

    @classmethod
    def deterministic(
        cls,
        name: str,
        *,
        lower_bound: float | None = 0.0,
        upper_bound: float | None = None,
        description: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> InteractionSpec:
        """Construct an interaction from a deterministic parameter reference."""
        return cls(
            value=DeterministicRef(name=name),
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            description=description,
            metadata=metadata or {},
        )

    @property
    def dependencies(self) -> set[str]:
        """All parameter-like dependencies."""
        return self.value.dependencies()

    @property
    def parameter_dependencies(self) -> set[str]:
        return self.value.parameter_dependencies()

    @property
    def deterministic_dependencies(self) -> set[str]:
        return self.value.deterministic_dependencies()

    @property
    def data_dependencies(self) -> set[str]:
        return self.value.data_dependencies()

    def evaluate(
        self,
        context: dict[str, Any] | None = None,
        data: Any | None = None,
    ) -> Any:
        """Evaluate the interaction value against a runtime parameter context."""
        return self.value.evaluate(
            context=context,
            data=data,
        )
