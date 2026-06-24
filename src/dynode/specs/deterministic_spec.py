from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from .value_spec import DeterministicExpression


class DeterministicSpec(BaseModel):
    """
    Declarative deterministic parameter.

    Example
    -------
    beta = r0 / infectious_period
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )

    name: str = Field(
        description="Name of the deterministic parameter.",
    )

    expression: DeterministicExpression = Field(
        description="Expression used to compute the deterministic parameter.",
    )

    description: str | None = Field(
        default=None,
        description="Optional human-readable description.",
    )

    @property
    def dependencies(self) -> set[str]:
        return self.expression.dependencies()

    @model_validator(mode="after")
    def validate_not_self_referential(self) -> Self:
        if self.name in self.dependencies:
            raise ValueError(
                f"Deterministic parameter {self.name!r} depends on itself."
            )

        return self

    def evaluate(self, context: dict[str, Any]) -> Any:
        """
        Evaluate the deterministic value against an already-sampled/resolved
        parameter context.
        """
        return self.expression.evaluate(context)

    def resolve(self, context: dict[str, Any]) -> Any:
        """
        Alias for evaluate(), useful if the rest of your framework already uses
        resolve terminology.
        """
        return self.evaluate(context)
