from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class StateBuilderOptions:
    """
    Options controlling initial-state construction.

    Shape validation is safe and useful. Finite/nonnegative checks should
    usually stay off during NumPyro/JAX tracing because they may require
    converting traced values to Python booleans.
    """

    validate_dependencies: bool = True
    validate_shapes: bool = True

    dtype: Any | None = None

    validate_finite: bool = False
    validate_nonnegative: bool = False
    strict_static_value_validation: bool = False
