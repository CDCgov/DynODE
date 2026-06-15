from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import date
from typing import Annotated, Any, Literal

import jax.numpy as jnp
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self


class ValueSpec(BaseModel):
    """
    Base class for serializable values used across the framework.

    Used by:
    - DistributionSpec
    - DeterministicSpec
    - InitializerSpec
    - InteractionSpec
    - StrainSpec
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )

    type: str

    def dependencies(self) -> set[str]:
        """
        All parameter-like dependencies.

        This includes sampled parameter refs and deterministic refs.
        """
        return self.parameter_dependencies() | self.deterministic_dependencies()

    def parameter_dependencies(self) -> set[str]:
        return set()

    def deterministic_dependencies(self) -> set[str]:
        return set()

    def data_dependencies(self) -> set[str]:
        return set()

    def evaluate(
        self,
        context: Mapping[str, Any] | None = None,
        data: Any | None = None,
    ) -> Any:
        raise NotImplementedError

class ConstantValueSpec(ValueSpec):
    """
    Literal scalar or array-like value.

    Examples
    --------
    1.0
    [1.0, 2.0, 3.0]
    [[1.0, 2.0], [3.0, 4.0]]
    """

    type: Literal["constant"] = "constant"

    value: Any = Field(
        description="Scalar or rectangular nested numeric list."
    )

    @model_validator(mode="after")
    def validate_constant_value(self) -> Self:
        self._validate_payload(self.value)
        return self

    def evaluate(
        self,
        context: Mapping[str, Any] | None = None,
        data: Any | None = None,
    ) -> Any:
        if isinstance(self.value, list):
            return jnp.asarray(self.value)

        return self.value

    @classmethod
    def _validate_payload(cls, value: Any) -> None:
        if isinstance(value, bool):
            return

        if isinstance(value, (int, float, date)):
            return

        if isinstance(value, list):
            for item in value:
                cls._validate_payload(item)

            try:
                array = np.asarray(value)
            except Exception as exc:
                raise ValueError(
                    "Constant value must be array-like if provided as a list."
                ) from exc

            if array.dtype == object:
                raise ValueError(
                    "Constant value appears to be ragged. "
                    "Use rectangular nested lists."
                )

            return

        raise TypeError(
            "Constant values must be int, float, bool, date, or nested numeric lists. "
            f"Got {type(value).__name__}."
        )

class ParamRef(ValueSpec):
    """
    Reference to a sampled parameter.

    Example
    -------
    {"type": "param_ref", "name": "r0"}
    """

    type: Literal["param_ref"] = "param_ref"

    name: str

    def parameter_dependencies(self) -> set[str]:
        return {self.name}

    def evaluate(
        self,
        context: Mapping[str, Any] | None = None,
        data: Any | None = None,
    ) -> Any:
        if context is None:
            raise ValueError(
                f"Cannot resolve parameter reference {self.name!r} without context."
            )

        try:
            return context[self.name]
        except KeyError as exc:
            raise KeyError(
                f"Parameter {self.name!r} was not found in context. "
                f"Available values are: {sorted(context)}."
            ) from exc

class DeterministicRef(ValueSpec):
    """
    Reference to an already-resolved deterministic parameter.
    """

    type: Literal["deterministic_ref"] = "deterministic_ref"

    name: str

    def deterministic_dependencies(self) -> set[str]:
        return {self.name}

    def evaluate(
        self,
        context: Mapping[str, Any] | None = None,
        data: Any | None = None,
    ) -> Any:
        if context is None:
            raise ValueError(
                f"Cannot resolve deterministic reference {self.name!r} without context."
            )

        try:
            return context[self.name]
        except KeyError as exc:
            raise KeyError(
                f"Deterministic parameter {self.name!r} was not found in context. "
                f"Available values are: {sorted(context)}."
            ) from exc

class DataRef(ValueSpec):
    """
    Reference to observed data.

    Useful for initializers or observation-related calculations.
    """

    type: Literal["data_ref"] = "data_ref"

    name: str
    index: int | None = None

    def data_dependencies(self) -> set[str]:
        return {self.name}

    def evaluate(
        self,
        context: Mapping[str, Any] | None = None,
        data: Any | None = None,
    ) -> Any:
        if data is None:
            raise ValueError(
                f"Cannot resolve data reference {self.name!r} without data."
            )

        value = self._lookup_data_value(data)

        if self.index is not None:
            value = value[self.index]

        return value

    def _lookup_data_value(self, data: Any) -> Any:
        if hasattr(data, "get_observation"):
            observation = data.get_observation(self.name)

            if hasattr(observation, "as_jax"):
                return observation.as_jax()

            return observation.values

        if isinstance(data, Mapping):
            if self.name in data:
                return data[self.name]

            observations = data.get("observations")
            if isinstance(observations, Mapping) and self.name in observations:
                return observations[self.name]

        raise KeyError(
            f"Could not find data reference {self.name!r}."
        )

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
                raise ValueError("clip requires exactly one positional argument.")

            if "min" not in self.kwargs or "max" not in self.kwargs:
                raise ValueError("clip requires keyword arguments 'min' and 'max'.")

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
        args = [
            arg.evaluate(context=context, data=data)
            for arg in self.args
        ]

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

def as_value_spec(value: Any) -> Any:
    """
    Convert raw Python values into ConstantValueSpec-compatible dictionaries.

    This lets you write:

        loc=0.0

    instead of:

        loc={"type": "constant", "value": 0.0}
    """
    if isinstance(value, ValueSpec):
        return value

    if isinstance(value, dict) and "type" in value:
        return value

    return {
        "type": "constant",
        "value": value,
    }


def coerce_value_fields(data: Any, field_names: Iterable[str]) -> Any:
    """
    Helper for model_validator(mode='before').

    Example
    -------
    class NormalSpec(...):
        loc: ValueExpression
        scale: ValueExpression

        @model_validator(mode="before")
        @classmethod
        def coerce_values(cls, data):
            return coerce_value_fields(data, ("loc", "scale"))
    """
    if not isinstance(data, dict):
        return data

    data = dict(data)

    for field_name in field_names:
        if field_name in data and data[field_name] is not None:
            data[field_name] = as_value_spec(data[field_name])

    return data

ValueExpression = Annotated[
    ConstantValueSpec
    | ParamRef
    | DeterministicRef
    | DataRef
    | UnaryValueSpec
    | BinaryValueSpec
    | FunctionValueSpec,
    Field(discriminator="type"),
]

for _model in (
    UnaryValueSpec,
    BinaryValueSpec,
    FunctionValueSpec,
):
    _model.model_rebuild(
        _types_namespace={"ValueExpression": ValueExpression}
    )

DistributionValue = ValueExpression
DeterministicExpression = ValueExpression
InitializerValue = ValueExpression
InteractionValue = ValueExpression
ParameterValue = ValueExpression
