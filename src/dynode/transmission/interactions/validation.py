from __future__ import annotations

from typing import Any

import numpy as np

from dynode.value.constant import ConstantValueSpec


def validate_bounds_are_ordered(
    *,
    lower_bound: float | None,
    upper_bound: float | None,
) -> None:
    if (
        lower_bound is not None
        and upper_bound is not None
        and upper_bound < lower_bound
    ):
        raise ValueError(
            "InteractionSpec upper_bound must be greater than or equal "
            f"to lower_bound. Got lower_bound={lower_bound}, "
            f"upper_bound={upper_bound}."
        )


def validate_constant_value_bounds(
    *,
    value: Any,
    lower_bound: float | None,
    upper_bound: float | None,
) -> None:
    """Validate bounds only when the interaction value is a constant."""
    if not isinstance(value, ConstantValueSpec):
        return

    array = constant_numeric_array(value)

    if lower_bound is not None and np.any(array < lower_bound):
        raise ValueError(
            f"Constant interaction value must be >= {lower_bound}."
        )

    if upper_bound is not None and np.any(array > upper_bound):
        raise ValueError(
            f"Constant interaction value must be <= {upper_bound}."
        )


def validate_no_data_dependencies(data_dependencies: set[str]) -> None:
    """Reject observed-data dependencies in interaction specs."""
    if data_dependencies:
        raise ValueError(
            "InteractionSpec cannot depend on observed data. "
            f"Data dependencies: {sorted(data_dependencies)}."
        )


def constant_numeric_array(value: ConstantValueSpec) -> np.ndarray:
    raw = value.value

    if isinstance(raw, bool):
        raise ValueError("Interaction value must be numeric, not bool.")

    try:
        return np.asarray(raw, dtype=float)
    except Exception as exc:
        raise ValueError("Interaction value must be numeric.") from exc
