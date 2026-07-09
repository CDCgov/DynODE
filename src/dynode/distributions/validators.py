from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Any

import numpy as np

from dynode.value.constant import ConstantValueSpec

if TYPE_CHECKING:
    from dynode.value.unions import DistributionValue


def _contains_bool_or_date(value: Any) -> bool:
    if isinstance(value, (bool, date)):
        return True

    if isinstance(value, list):
        return any(_contains_bool_or_date(item) for item in value)

    return False


def _constant_as_numeric_array(
    value: ConstantValueSpec,
    field_name: str,
) -> np.ndarray:
    """
    Convert a ConstantValueSpec payload into a numeric NumPy array for validation.

    Distribution parameters should be numeric, not bool/date.
    """
    raw = value.value

    if _contains_bool_or_date(raw):
        raise ValueError(f"{field_name} must be numeric, not bool/date.")

    try:
        return np.asarray(raw, dtype=float)
    except Exception as exc:
        raise ValueError(f"{field_name} must be numeric.") from exc


def _validate_constant_numeric(
    value: DistributionValue,
    field_name: str,
) -> None:
    """
    Validate that a constant distribution parameter is numeric.

    ParamRef / DeterministicRef / expressions are checked at runtime.
    """
    if not isinstance(value, ConstantValueSpec):
        return

    _constant_as_numeric_array(value, field_name)


def _validate_constant_positive(
    value: DistributionValue,
    field_name: str,
) -> None:
    """
    Validate that a constant distribution parameter is strictly positive.

    ParamRef / DeterministicRef / expressions are checked at runtime.
    """
    if not isinstance(value, ConstantValueSpec):
        return

    arr = _constant_as_numeric_array(value, field_name)

    if np.any(arr <= 0):
        raise ValueError(f"{field_name} must be positive.")


def _validate_constant_bounds(
    low: DistributionValue | None,
    high: DistributionValue | None,
) -> None:
    """
    Validate high > low when both are constant values.
    """
    if low is None or high is None:
        return

    if not isinstance(low, ConstantValueSpec):
        return

    if not isinstance(high, ConstantValueSpec):
        return

    low_arr = _constant_as_numeric_array(low, "TruncatedNormal low")
    high_arr = _constant_as_numeric_array(high, "TruncatedNormal high")

    try:
        invalid = np.any(high_arr <= low_arr)
    except ValueError as exc:
        raise ValueError(
            "TruncatedNormal low and high constants are not broadcast-compatible."
        ) from exc

    if invalid:
        raise ValueError("TruncatedNormal high must be greater than low.")
