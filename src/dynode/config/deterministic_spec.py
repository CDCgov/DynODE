from __future__ import annotations

from typing import Annotated, Any, Literal

import jax.numpy as jnp
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self


class ExpressionSpec(BaseModel):
    """
    Base class for deterministic expressions.

    Expressions should be:
    - serializable
    - validated by Pydantic
    - evaluable against a runtime parameter context
    - able to report their dependencies
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )

    type: str

    def dependencies(self) -> set[str]:
        raise NotImplementedError

    def evaluate(self, context: dict[str, Any]) -> Any:
        raise NotImplementedError


class ConstantExpressionSpec(ExpressionSpec):
    type: Literal["constant"] = "constant"

    value: int | float | bool | list[int] | list[float]

    def dependencies(self) -> set[str]:
        return set()

    def evaluate(self, context: dict[str, Any]) -> Any:
        if isinstance(self.value, list):
            return jnp.asarray(self.value)
        return self.value


class ParamRef(ExpressionSpec):
    """
    Reference to a sampled parameter.

    Example
    -------
    beta = ParamRef(name="beta")
    """

    type: Literal["param_ref"] = "param_ref"

    name: str

    def dependencies(self) -> set[str]:
        return {self.name}

    def evaluate(self, context: dict[str, Any]) -> Any:
        try:
            return context[self.name]
        except KeyError as exc:
            raise KeyError(
                f"Parameter {self.name!r} was not found in runtime context. "
                f"Available values are: {sorted(context)}."
            ) from exc


class DeterministicRef(ExpressionSpec):
    """
    Reference to another deterministic parameter.

    This is useful for validating dependency order.
    """

    type: Literal["deterministic_ref"] = "deterministic_ref"

    name: str

    def dependencies(self) -> set[str]:
        return {self.name}

    def evaluate(self, context: dict[str, Any]) -> Any:
        try:
            return context[self.name]
        except KeyError as exc:
            raise KeyError(
                f"Deterministic parameter {self.name!r} was not found in "
                f"runtime context. Available values are: {sorted(context)}."
            ) from exc


class UnaryExpressionSpec(ExpressionSpec):
    type: Literal["unary"] = "unary"

    op: Literal[
        "neg",
        "exp",
        "log",
        "sqrt",
        "log1p",
        "sigmoid",
    ]

    arg: Expression

    def dependencies(self) -> set[str]:
        return self.arg.dependencies()

    def evaluate(self, context: dict[str, Any]) -> Any:
        value = self.arg.evaluate(context)

        if self.op == "neg":
            return -value

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


class BinaryExpressionSpec(ExpressionSpec):
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

    left: Expression
    right: Expression

    def dependencies(self) -> set[str]:
        return self.left.dependencies() | self.right.dependencies()

    def evaluate(self, context: dict[str, Any]) -> Any:
        left = self.left.evaluate(context)
        right = self.right.evaluate(context)

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


class FunctionExpressionSpec(ExpressionSpec):
    """
    Function-style expression for common JAX-safe operations.

    Use this when the operation is naturally variadic or has named behavior.
    """

    type: Literal["function"] = "function"

    function: Literal[
        "sum",
        "mean",
        "prod",
        "clip",
    ]

    args: tuple[Expression, ...]
    kwargs: dict[str, Expression] = Field(default_factory=dict)

    def dependencies(self) -> set[str]:
        deps: set[str] = set()

        for arg in self.args:
            deps |= arg.dependencies()

        for value in self.kwargs.values():
            deps |= value.dependencies()

        return deps

    def evaluate(self, context: dict[str, Any]) -> Any:
        args = [arg.evaluate(context) for arg in self.args]
        kwargs = {
            key: value.evaluate(context) for key, value in self.kwargs.items()
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
            if len(args) != 1:
                raise ValueError(
                    "clip expects exactly one positional argument."
                )

            if "min" not in kwargs or "max" not in kwargs:
                raise ValueError(
                    "clip requires keyword expressions 'min' and 'max'."
                )

            return jnp.clip(args[0], kwargs["min"], kwargs["max"])

        raise ValueError(
            f"Unsupported function expression: {self.function!r}."
        )


Expression = Annotated[
    ConstantExpressionSpec
    | ParamRef
    | DeterministicRef
    | UnaryExpressionSpec
    | BinaryExpressionSpec
    | FunctionExpressionSpec,
    Field(discriminator="type"),
]


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

    expression: Expression = Field(
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
