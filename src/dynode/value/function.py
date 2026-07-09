from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal

import jax.numpy as jnp
from pydantic import Field, model_validator
from typing_extensions import Self

from .base import ValueSpec

if TYPE_CHECKING:
    from .unions import ValueExpression


class FunctionValueSpec(ValueSpec):
    type: Literal["function"] = "function"

    function: Literal[
        "sum",
        "mean",
        "prod",
        "clip",
        "stack",
        "concatenate",
    ]

    args: tuple[ValueExpression, ...] = Field(default_factory=tuple)
    kwargs: dict[str, ValueExpression] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_function_shape(self) -> Self:
        if self.function in {"sum", "mean", "prod", "stack", "concatenate"}:
            if not self.args:
                raise ValueError(
                    f"Function {self.function!r} requires at least one argument."
                )

        if self.function == "clip":
            if len(self.args) != 1:
                raise ValueError(
                    "clip requires exactly one positional argument."
                )

            if "min" not in self.kwargs or "max" not in self.kwargs:
                raise ValueError(
                    "clip requires keyword arguments 'min' and 'max'."
                )

        return self

    def parameter_dependencies(self) -> set[str]:
        deps: set[str] = set()

        for arg in self.args:
            deps |= arg.parameter_dependencies()

        for value in self.kwargs.values():
            deps |= value.parameter_dependencies()

        return deps

    def deterministic_dependencies(self) -> set[str]:
        deps: set[str] = set()

        for arg in self.args:
            deps |= arg.deterministic_dependencies()

        for value in self.kwargs.values():
            deps |= value.deterministic_dependencies()

        return deps

    def data_dependencies(self) -> set[str]:
        deps: set[str] = set()

        for arg in self.args:
            deps |= arg.data_dependencies()

        for value in self.kwargs.values():
            deps |= value.data_dependencies()

        return deps

    def evaluate(
        self,
        context: Mapping[str, Any] | None = None,
        data: Any | None = None,
    ) -> Any:
        args = [arg.evaluate(context=context, data=data) for arg in self.args]

        kwargs = {
            key: value.evaluate(context=context, data=data)
            for key, value in self.kwargs.items()
        }

        if self.function == "sum":
            return sum(args)

        if self.function == "mean":
            return sum(args) / len(args)

        if self.function == "prod":
            result = 1
            for arg in args:
                result = result * arg
            return result

        if self.function == "clip":
            return jnp.clip(args[0], kwargs["min"], kwargs["max"])

        if self.function == "stack":
            axis = int(kwargs.get("axis", 0))
            return jnp.stack(args, axis=axis)

        if self.function == "concatenate":
            axis = int(kwargs.get("axis", 0))
            return jnp.concatenate(args, axis=axis)

        raise ValueError(f"Unsupported function: {self.function!r}.")
