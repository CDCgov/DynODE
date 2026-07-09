from __future__ import annotations

from typing import Any, Literal

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from dynode.typing import DynodeName


class DataFieldSpec(BaseModel):
    """Specification for an observed array, covariate, index array, or metadata field."""

    model_config = ConfigDict(
        extra="forbid", frozen=True, arbitrary_types_allowed=True
    )

    name: DynodeName
    kind: Literal["observed", "covariate", "index", "metadata"] = "observed"
    dims: tuple[str, ...] = Field(default_factory=tuple)
    dtype: Literal["float", "int", "bool", "date", "string", "object"] = (
        "float"
    )
    shape: tuple[int | None, ...] | None = None
    required: bool = True
    allow_missing: bool = False
    description: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)

    def validate_value(self, value: Any) -> None:
        if value is None:
            if self.required and not self.allow_missing:
                raise ValueError(
                    f"Data field {self.name!r} is required but value is None."
                )
            return
        if self.kind == "metadata" or self.dtype in {
            "date",
            "string",
            "object",
        }:
            return
        arr = np.asarray(value)
        if self.shape is not None:
            if len(arr.shape) != len(self.shape):
                raise ValueError(
                    f"Data field {self.name!r} expected ndim {len(self.shape)}, got shape {arr.shape}."
                )
            for idx, (actual, expected) in enumerate(
                zip(arr.shape, self.shape)
            ):
                if expected is not None and actual != expected:
                    raise ValueError(
                        f"Data field {self.name!r} axis {idx} expected size {expected}, got {actual}."
                    )
        if self.dtype == "float" and not np.issubdtype(arr.dtype, np.number):
            raise ValueError(f"Data field {self.name!r} must be numeric.")
        if self.dtype == "int" and not np.issubdtype(arr.dtype, np.integer):
            raise ValueError(
                f"Data field {self.name!r} must be integer typed."
            )
