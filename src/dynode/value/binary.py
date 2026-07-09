from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal

import jax.numpy as jnp

from .base import ValueSpec

if TYPE_CHECKING:
    from .unions import ValueExpression


class BinaryValueSpec(ValueSpec):
    type: Literal["binary"] = "binary"

    op: Literal[
        "add",
        "sub",
        "mul",
        "div",
        "pow",
        "minimum",
        "maximum",
    ]

    left: ValueExpression
    right: ValueExpression

    def parameter_dependencies(self) -> set[str]:
        return (
            self.left.parameter_dependencies()
            | self.right.parameter_dependencies()
        )

    def deterministic_dependencies(self) -> set[str]:
        return (
            self.left.deterministic_dependencies()
            | self.right.deterministic_dependencies()
        )

    def data_dependencies(self) -> set[str]:
        return self.left.data_dependencies() | self.right.data_dependencies()

    def evaluate(
        self,
        context: Mapping[str, Any] | None = None,
        data: Any | None = None,
    ) -> Any:
        left = self.left.evaluate(context=context, data=data)
        right = self.right.evaluate(context=context, data=data)

        if self.op == "add":
            return left + right

        if self.op == "sub":
            return left - right

        if self.op == "mul":
            return left * right

        if self.op == "div":
            return left / right

        if self.op == "pow":
            return left**right

        if self.op == "minimum":
            return jnp.minimum(left, right)

        if self.op == "maximum":
            return jnp.maximum(left, right)

        raise ValueError(f"Unsupported binary operation: {self.op!r}.")
