from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .base import ValueSpec


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
