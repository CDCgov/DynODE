from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .types import RHSCallStyle, RHSStateFormat


@dataclass(frozen=True, slots=True)
class OdeSolverOptions:
    """
    Runtime options for wrapping and validating the ODE solve.

    This class intentionally does not duplicate SolverSpec. SolverSpec owns:
    - solver method
    - dt0
    - step-size controller
    - save_at
    - max_steps
    - throw
    - jump_ts / step_ts handling
    """

    rhs_state_format: RHSStateFormat = "flat"
    rhs_call_style: RHSCallStyle = "auto"

    validate_y0: bool = True
    validate_rhs_output_shape: bool = True
    validate_solution_shape: bool = True

    y0_dtype: Any | None = None

    rhs_extra: Mapping[str, Any] = field(default_factory=dict)
