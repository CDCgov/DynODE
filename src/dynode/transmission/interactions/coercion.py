from __future__ import annotations

from typing import Any

from dynode.value.coercion import as_value_spec


def coerce_interaction_data(data: Any) -> Any:
    """Coerce compact interaction forms into InteractionSpec-compatible data."""
    if isinstance(data, dict):
        data = dict(data)

        if "value" in data:
            data["value"] = as_value_spec(data["value"])
            return data

        # If the dict looks like a ValueSpec, wrap it as the interaction value.
        if "type" in data:
            return {
                "value": data,
            }

        return data

    return {
        "value": as_value_spec(data),
    }
