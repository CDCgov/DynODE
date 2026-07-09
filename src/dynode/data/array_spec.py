from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from dynode.value.coercion import as_value_spec

if TYPE_CHECKING:
    from dynode.value.unions import ValueExpression


class ArraySpec(BaseModel):
    """Value expression with named array dimensions."""

    model_config = ConfigDict(
        extra="forbid", frozen=True, arbitrary_types_allowed=True
    )

    value: ValueExpression
    dims: tuple[str, ...] = Field(default_factory=tuple)
    shape: tuple[int | None, ...] | None = None
    description: str | None = None

    @model_validator(mode="before")
    @classmethod
    def coerce_value(cls, data: Any) -> Any:
        if isinstance(data, dict) and "value" in data:
            data = dict(data)
            data["value"] = as_value_spec(data["value"])
        return data

    @model_validator(mode="after")
    def validate_shape_rank(self) -> Self:
        if self.shape is not None and len(self.shape) != len(self.dims):
            raise ValueError(
                "ArraySpec.shape and ArraySpec.dims must have the same rank."
            )
        return self

    def dependencies(self) -> set[str]:
        return self.value.dependencies()

    def data_dependencies(self) -> set[str]:
        return self.value.data_dependencies()

    def evaluate(
        self, context: dict[str, Any] | None = None, data: Any | None = None
    ) -> Any:
        result = self.value.evaluate(context=context, data=data)
        if self.shape is not None:
            arr = np.asarray(result)
            if len(arr.shape) != len(self.shape):
                raise ValueError(
                    f"ArraySpec expected rank {len(self.shape)}, got shape {arr.shape}."
                )
            for axis, (actual, expected) in enumerate(
                zip(arr.shape, self.shape)
            ):
                if expected is not None and actual != expected:
                    raise ValueError(
                        f"ArraySpec axis {axis} expected size {expected}, got {actual}."
                    )
        return result
