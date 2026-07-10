from __future__ import annotations

from typing import Any

from .base import BinSpec


def as_bin_spec(value: Any) -> Any:
    """
    Coerce compact bin inputs into explicit bin specs.

    Examples
    --------
    {"name": "none"}
        -> {"type": "generic", "name": "none"}

    "none"
        -> {"type": "generic", "name": "none"}

    {"type": "age", "min_value": 0, "max_value": 4}
        -> unchanged
    """
    if isinstance(value, BinSpec):
        return value

    if isinstance(value, str):
        return {
            "type": "generic",
            "name": value,
        }

    if isinstance(value, dict):
        value = dict(value)

        if "type" in value:
            return value

        if "min_value" in value and "max_value" in value:
            return {
                "type": "discretized_positive_int",
                **value,
            }

        if "name" in value:
            return {
                "type": "generic",
                **value,
            }

    return value


def coerce_bin_specs(values: Any) -> Any:
    """
    Coerce a list/tuple of compact bin configs into explicit bin specs.
    """
    if values is None:
        return values

    return tuple(as_bin_spec(value) for value in values)
