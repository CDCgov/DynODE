from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

if TYPE_CHECKING:
    from dynode.value.unions import DeterministicExpression


class DeterministicSpec(BaseModel):
    """Declarative deterministic parameter."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )

    name: str = Field(description="Name of the deterministic parameter.")
    expression: DeterministicExpression = Field(
        description="Expression used to compute the deterministic parameter."
    )
    description: str | None = None

    @property
    def dependencies(self) -> set[str]:
        return self.expression.dependencies()

    @property
    def parameter_dependencies(self) -> set[str]:
        return self.expression.parameter_dependencies()

    @property
    def deterministic_dependencies(self) -> set[str]:
        return self.expression.deterministic_dependencies()

    @property
    def data_dependencies(self) -> set[str]:
        return self.expression.data_dependencies()

    @model_validator(mode="after")
    def validate_not_self_referential(self) -> Self:
        if self.name in self.dependencies:
            raise ValueError(
                f"Deterministic parameter {self.name!r} depends on itself."
            )
        return self

    def evaluate(
        self,
        context: dict[str, Any],
        data: Any | None = None,
    ) -> Any:
        return self.expression.evaluate(context=context, data=data)

    def resolve(
        self,
        context: dict[str, Any],
        data: Any | None = None,
    ) -> Any:
        return self.evaluate(context=context, data=data)
