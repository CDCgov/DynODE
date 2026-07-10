from __future__ import annotations

from typing import Any

from dynode.runtime.layout.runtime_model import RuntimeModel

from .errors import StateBuilderError


def state_view(
    *,
    runtime: RuntimeModel,
    flat_state: Any,
    compartment_name: str,
) -> Any:
    """
    Return one compartment array from a flat state vector.
    """
    try:
        return runtime.state_layout.view(
            flat_state,
            compartment_name,
        )
    except Exception as exc:
        raise StateBuilderError(
            f"Could not extract compartment {compartment_name!r} from flat state."
        ) from exc


def replace_state_view(
    *,
    runtime: RuntimeModel,
    flat_state: Any,
    compartment_name: str,
    value: Any,
    allow_broadcast: bool = False,
) -> Any:
    """
    Replace one compartment in a flat state vector.
    """
    try:
        return runtime.state_layout.replace(
            flat_state,
            compartment_name,
            value,
            allow_broadcast=allow_broadcast,
        )
    except Exception as exc:
        raise StateBuilderError(
            f"Could not replace compartment {compartment_name!r} in flat state."
        ) from exc