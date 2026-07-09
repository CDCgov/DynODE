from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal

import jax.numpy as jnp

from .base import ValueSpec

if TYPE_CHECKING:
    from .unions import ValueExpression


class UnaryValueSpec(ValueSpec):
    type: Literal["unary"] = "unary"

    op: Literal[
        "neg",
        "abs",
        "exp",
        "log",
        "sqrt",
        "log1p",
        "sigmoid",
    ]

    arg: ValueExpression

    def parameter_dependencies(self) -> set[str]:
        return self.arg.parameter_dependencies()

    def deterministic_dependencies(self) -> set[str]:
        return self.arg.deterministic_dependencies()

    def data_dependencies(self) -> set[str]:
        return self.arg.data_dependencies()

    def evaluate(
        self,
        context: Mapping[str, Any] | None = None,
        data: Any | None = None,
    ) -> Any:
        value = self.arg.evaluate(context=context, data=data)

        if self.op == "neg":
            return -value

        if self.op == "abs":
            return jnp.abs(value)

        if self.op == "exp":
            return jnp.exp(value)

        if self.op == "log":
            return jnp.log(value)

        if self.op == "sqrt":
            return jnp.sqrt(value)

        if self.op == "log1p":
            return jnp.log1p(value)

        if self.op == "sigmoid":
            return 1 / (1 + jnp.exp(-value))

        raise ValueError(f"Unsupported unary operation: {self.op!r}.")
