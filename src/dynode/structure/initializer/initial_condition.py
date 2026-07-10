from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dynode.typing import DynodeName
from dynode.value.coercion import as_value_spec
from dynode.value.constant import ConstantValueSpec
from dynode.value.unions import InitializerValue


class CompartmentInitialConditionSpec(BaseModel):
    """
    Initial condition for one compartment.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )

    compartment_name: DynodeName = Field(
        description="Name of the compartment to initialize."
    )

    value: InitializerValue = Field(
        default_factory=lambda: ConstantValueSpec(value=0.0),
        description=(
            "Initial value for this compartment. Scalars are broadcast to the "
            "full compartment shape."
        ),
    )

    allow_broadcast: bool = Field(
        default=True,
        description="Whether scalar or lower-dimensional values may be broadcast.",
    )

    description: str | None = Field(
        default=None,
        description="Optional human-readable description.",
    )

    @model_validator(mode="before")
    @classmethod
    def coerce_value(cls, data: Any) -> Any:
        if isinstance(data, dict) and "value" in data:
            data = dict(data)
            data["value"] = as_value_spec(data["value"])

        return data

    @property
    def dependencies(self) -> set[str]:
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
        return self.value.evaluate(
            context=context,
            data=data,
        )
