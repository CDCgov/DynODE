from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from dynode.value.coercion import as_value_spec
from dynode.value.constant import ConstantValueSpec
from dynode.value.references import DeterministicRef, ParamRef

if TYPE_CHECKING:
    from dynode.value.unions import InteractionValue


class InteractionSpec(BaseModel):
    """
    Declarative strain-interaction specification.

    This represents one entry in the strain interaction matrix:

        interaction[source_strain, target_strain]

    Examples
    --------
    Fixed interaction:

        InteractionSpec.fixed(1.0)

    Parameter reference:

        InteractionSpec.parameter("crossimmunity")

    Deterministic reference:

        InteractionSpec.deterministic("crossimmunity")

    Expression:

        InteractionSpec(
            value={
                "type": "binary",
                "op": "mul",
                "left": {"type": "param_ref", "name": "base_crossimmunity"},
                "right": {"type": "constant", "value": 0.8},
            }
        )
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
        """
        Allows compact forms.

        These are all valid:

            InteractionSpec(value=0.5)

            {"value": 0.5}

            0.5

            {"type": "param_ref", "name": "crossimmunity"}

        The last form is interpreted as the value expression itself.
        """
        if isinstance(data, InteractionSpec):
            return data

        if isinstance(data, dict):
            data = dict(data)

            if "value" in data:
                data["value"] = as_value_spec(data["value"])
                return data

            # If the dict looks like a ValueSpec, wrap it as the interaction value.
            if "type" in data:
                return {
                    "value": data,
                }

            return data

        return {
            "value": as_value_spec(data),
        }

    @model_validator(mode="after")
    def validate_interaction(self) -> Self:
        self._validate_bounds_are_ordered()
        self._validate_constant_value_bounds()
        self._validate_no_data_dependencies()
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
        """
        Construct a fixed constant interaction.

        Example
        -------
        InteractionSpec.fixed(1.0)
        """
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
        """
        Construct an interaction from a sampled parameter reference.

        Example
        -------
        InteractionSpec.parameter("crossimmunity")
        """
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
        """
        Construct an interaction from a deterministic parameter reference.

        Example
        -------
        InteractionSpec.deterministic("crossimmunity")
        """
        return cls(
            value=DeterministicRef(name=name),
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            description=description,
            metadata=metadata or {},
        )

    @property
    def dependencies(self) -> set[str]:
        """
        All parameter-like dependencies.

        Includes sampled parameter refs and deterministic refs.
        """
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
        """
        Evaluate the interaction value against a runtime parameter context.

        This intentionally does not perform Python-side bounds checks on
        non-constant runtime values, because those values may be JAX tracers
        inside NumPyro/JAX execution.
        """
        return self.value.evaluate(
            context=context,
            data=data,
        )

    def _validate_bounds_are_ordered(self) -> None:
        if (
            self.lower_bound is not None
            and self.upper_bound is not None
            and self.upper_bound < self.lower_bound
        ):
            raise ValueError(
                "InteractionSpec upper_bound must be greater than or equal "
                f"to lower_bound. Got lower_bound={self.lower_bound}, "
                f"upper_bound={self.upper_bound}."
            )

    def _validate_constant_value_bounds(self) -> None:
        """
        Validate bounds only when the interaction value is a constant.

        Parameter references and expressions are validated by their priors or
        deterministic definitions.
        """
        if not isinstance(self.value, ConstantValueSpec):
            return

        array = self._constant_numeric_array(self.value)

        if self.lower_bound is not None and np.any(array < self.lower_bound):
            raise ValueError(
                f"Constant interaction value must be >= {self.lower_bound}."
            )

        if self.upper_bound is not None and np.any(array > self.upper_bound):
            raise ValueError(
                f"Constant interaction value must be <= {self.upper_bound}."
            )

    def _validate_no_data_dependencies(self) -> None:
        """
        Strain interactions should generally not depend on observed data.

        If your framework later intentionally supports data-derived interaction
        values, this validator can be relaxed.
        """
        data_deps = self.data_dependencies

        if data_deps:
            raise ValueError(
                "InteractionSpec cannot depend on observed data. "
                f"Data dependencies: {sorted(data_deps)}."
            )

    @staticmethod
    def _constant_numeric_array(value: ConstantValueSpec) -> np.ndarray:
        raw = value.value

        if isinstance(raw, bool):
            raise ValueError("Interaction value must be numeric, not bool.")

        try:
            return np.asarray(raw, dtype=float)
        except Exception as exc:
            raise ValueError("Interaction value must be numeric.") from exc
