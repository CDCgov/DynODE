from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from typing import Any, Literal

import jax.numpy as jnp
import numpy as np
from pydantic import Field, model_validator
from typing_extensions import Self

from .base import ValueSpec


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
