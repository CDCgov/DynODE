from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from dynode.value.coercion import as_value_spec

if TYPE_CHECKING:
    from dynode.value.unions import ValueExpression


class FunctionRef(BaseModel):
    """Serializable reference to a runtime function registry entry."""

    model_config = ConfigDict(
        extra="forbid", frozen=True, arbitrary_types_allowed=True
    )

    name: str
    kwargs: dict[str, ValueExpression] = Field(default_factory=dict)

    @classmethod
    def with_constants(cls, name: str, **kwargs: Any) -> "FunctionRef":
        return cls(
            name=name,
            kwargs={
                key: as_value_spec(value) for key, value in kwargs.items()
            },
        )

    def dependencies(self) -> set[str]:
        deps: set[str] = set()
        for value in self.kwargs.values():
            deps |= value.dependencies()
        return deps

    def data_dependencies(self) -> set[str]:
        deps: set[str] = set()
        for value in self.kwargs.values():
            deps |= value.data_dependencies()
        return deps

    def evaluate_kwargs(
        self, context: Mapping[str, Any] | None = None, data: Any | None = None
    ) -> dict[str, Any]:
        return {
            key: value.evaluate(context=context, data=data)
            for key, value in self.kwargs.items()
        }
