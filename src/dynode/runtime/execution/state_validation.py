from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp
import numpy as np

from dynode.runtime.layout.runtime_model import RuntimeModel

from .errors import StateBuilderError
from .state_options import StateBuilderOptions
from .state_utils import available_data_names, dependency_set
from .types import FlatState, ParameterMapping, StateDict


def normalize_state_dict(
    *,
    runtime: RuntimeModel,
    state: Mapping[str, Any],
    options: StateBuilderOptions | None = None,
) -> StateDict:
    """
    Normalize and validate a state dictionary against RuntimeModel.state_layout.

    InitializerSpec should already produce correctly shaped arrays. This function
    is a defensive runtime-layout check and optional dtype coercion point.
    """
    options = options or StateBuilderOptions()

    if not isinstance(state, Mapping):
        raise StateBuilderError(
            "Initial state must be a mapping from compartment name to value. "
            f"Got {type(state).__name__}."
        )

    expected_names = set(runtime.state_layout.compartment_names)
    supplied_names = set(state)

    missing = sorted(expected_names - supplied_names)
    extra = sorted(supplied_names - expected_names)

    if missing:
        raise StateBuilderError(
            f"Initial state is missing compartments: {missing}."
        )

    if extra:
        raise StateBuilderError(
            f"Initial state contains unknown compartments: {extra}. "
            f"Known compartments are: {runtime.state_layout.compartment_names}."
        )

    normalized: StateDict = {}

    for compartment in runtime.state_layout.compartments:
        value = jnp.asarray(state[compartment.name])

        if options.dtype is not None:
            value = value.astype(options.dtype)

        if options.validate_shapes:
            value_shape = tuple(value.shape)

            if value_shape != compartment.shape:
                raise StateBuilderError(
                    f"Initial state for compartment {compartment.name!r} has "
                    f"shape {value_shape}, but runtime layout expects "
                    f"{compartment.shape}."
                )

        normalized[compartment.name] = value

    validate_static_values_if_requested(
        values=normalized.values(),
        options=options,
        label="initial state",
    )

    return normalized


def validate_state_dict(
    *,
    runtime: RuntimeModel,
    state: Mapping[str, Any],
    options: StateBuilderOptions | None = None,
) -> None:
    """
    Validate a state dictionary against RuntimeModel.state_layout.
    """
    normalize_state_dict(
        runtime=runtime,
        state=state,
        options=options,
    )


def validate_flat_state(
    *,
    runtime: RuntimeModel,
    flat_state: Any,
    options: StateBuilderOptions | None = None,
) -> None:
    """
    Validate a flat state vector against RuntimeModel.state_layout.
    """
    options = options or StateBuilderOptions()

    if options.validate_shapes:
        try:
            runtime.state_layout.validate_flat_state(flat_state)
        except Exception as exc:
            raise StateBuilderError(
                "Flat initial state does not match RuntimeModel.state_layout."
            ) from exc

    validate_static_values_if_requested(
        values=(flat_state,),
        options=options,
        label="flat initial state",
    )


def flatten_state_dict(
    *,
    runtime: RuntimeModel,
    state: Mapping[str, Any],
    options: StateBuilderOptions | None = None,
) -> FlatState:
    """
    Normalize and flatten a compartment state dictionary.
    """
    options = options or StateBuilderOptions()

    normalized = normalize_state_dict(
        runtime=runtime,
        state=state,
        options=options,
    )

    flat_state = runtime.state_layout.flatten(
        normalized,
        allow_broadcast=False,
    )

    validate_flat_state(
        runtime=runtime,
        flat_state=flat_state,
        options=options,
    )

    return flat_state


def unflatten_state(
    *,
    runtime: RuntimeModel,
    flat_state: Any,
    options: StateBuilderOptions | None = None,
) -> StateDict:
    """
    Convert a flat state vector into a compartment dictionary.
    """
    options = options or StateBuilderOptions()

    validate_flat_state(
        runtime=runtime,
        flat_state=flat_state,
        options=options,
    )

    state = runtime.state_layout.unflatten(flat_state)

    return normalize_state_dict(
        runtime=runtime,
        state=state,
        options=options,
    )


def validate_initializer_context(
    *,
    runtime: RuntimeModel,
    params: ParameterMapping,
    data: Any | None = None,
) -> None:
    """
    Validate that InitializerSpec has the parameter and data values it needs.

    compile_model.py should already catch most of this statically. This remains
    useful as a runtime guard, especially when data is supplied dynamically.
    """
    initializer = runtime.initializer_spec

    parameter_deps = dependency_set(
        initializer,
        "dependencies",
    )

    missing_params = sorted(parameter_deps - set(params))

    if missing_params:
        raise StateBuilderError(
            "Initializer depends on parameters that are missing from the "
            f"parameter context: {missing_params}. Available parameters are: "
            f"{sorted(params)}."
        )

    data_deps = dependency_set(
        initializer,
        "data_dependencies",
    )

    if not data_deps:
        return

    if data is None:
        raise StateBuilderError(
            "Initializer has data dependencies but no data object was supplied. "
            f"Dependencies: {sorted(data_deps)}."
        )

    available_data = available_data_names(data)
    missing_data = sorted(data_deps - available_data)

    if missing_data:
        raise StateBuilderError(
            "Initializer depends on data series that are not available: "
            f"{missing_data}. Available data series are: {sorted(available_data)}."
        )


def validate_static_values_if_requested(
    *,
    values: Any,
    options: StateBuilderOptions,
    label: str,
) -> None:
    """
    Optional static validation for deterministic/unit-test contexts.

    Avoid enabling these checks during NumPyro/JAX tracing unless you are sure
    values are concrete arrays.
    """
    if not options.validate_finite and not options.validate_nonnegative:
        return

    for value in values:
        try:
            array = np.asarray(value)
        except Exception as exc:
            if options.strict_static_value_validation:
                raise StateBuilderError(
                    f"Could not convert {label} to a NumPy array for static "
                    "value validation."
                ) from exc

            continue

        if not np.issubdtype(array.dtype, np.number):
            if options.strict_static_value_validation:
                raise StateBuilderError(
                    f"Could not validate {label}; array is not numeric."
                )

            continue

        if options.validate_finite and not np.all(np.isfinite(array)):
            raise StateBuilderError(f"{label} contains non-finite values.")

        if options.validate_nonnegative and np.any(array < 0):
            raise StateBuilderError(f"{label} contains negative values.")
