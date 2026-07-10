from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dynode.runtime.context.experiment_runtime import ModelInstanceRuntime


def select_instance_data(
    instance: ModelInstanceRuntime,
    data: Mapping[str, Any] | None,
) -> Any:
    if data is None:
        return None

    if instance.key in data:
        return data[instance.key]

    try:
        numeric_key = int(instance.key)
    except (TypeError, ValueError):
        numeric_key = None

    if numeric_key is not None and numeric_key in data:
        return data[numeric_key]

    return data


def build_instance_data_bundle(
    *,
    instance: ModelInstanceRuntime,
    selected_data: Any,
) -> Any:
    static_context = dict(instance.static_context)

    if isinstance(selected_data, Mapping):
        data_bundle = dict(static_context)
        data_bundle.update(dict(selected_data))
        return data_bundle

    if selected_data is None:
        return dict(static_context)

    return selected_data


def build_initial_instance_context(
    *,
    shared_context: Mapping[str, Any],
    static_context: Mapping[str, Any],
) -> dict[str, Any]:
    initial_context = dict(shared_context)
    parameter_context = static_context.get("parameter_context", {})

    if isinstance(parameter_context, Mapping):
        initial_context.update(parameter_context)

    return initial_context


def build_local_context(
    *,
    full_context: Mapping[str, Any],
    shared_context: Mapping[str, Any],
    instance: ModelInstanceRuntime,
) -> dict[str, Any]:
    return {
        key: value
        for key, value in full_context.items()
        if key not in shared_context
        or key in instance.parameter_layout.resolved_name_set
    }
