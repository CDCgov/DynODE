from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np

from .runtime_model import RuntimeModel

StateDict = dict[str, Any]
FlatState = Any
ParameterContext = Mapping[str, Any]


class StateBuilderError(ValueError):
    """
    Raised when an initial state cannot be built from RuntimeModel and
    InitializerSpec.
    """


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


def build_initial_state(
    *,
    runtime: RuntimeModel,
    params: ParameterContext,
    data: Any | None = None,
    flat: bool = True,
    options: StateBuilderOptions | None = None,
) -> FlatState | StateDict:
    """
    Build the initial ODE state.

    Parameters
    ----------
    runtime:
        Compiled RuntimeModel.

    params:
        Full parameter context produced by parameter_sampling.py.

    data:
        Optional data object. Initializer values may use DataRef.

    flat:
        If True, return a flat state vector. If False, return a compartment dict.

    options:
        State-building options.
    """
    if flat:
        return build_initial_state_flat(
            runtime=runtime,
            params=params,
            data=data,
            options=options,
        )

    return build_initial_state_dict(
        runtime=runtime,
        params=params,
        data=data,
        options=options,
    )


def build_initial_state_dict(
    *,
    runtime: RuntimeModel,
    params: ParameterContext,
    data: Any | None = None,
    options: StateBuilderOptions | None = None,
) -> StateDict:
    """
    Build the initial state as a dictionary of compartment arrays.

    This delegates compartment-specific initialization to InitializerSpec.
    """
    options = options or StateBuilderOptions()

    if options.validate_dependencies:
        validate_initializer_context(
            runtime=runtime,
            params=params,
            data=data,
        )

    initializer = runtime.initializer_spec

    try:
        raw_state = initializer.build_dict(
            simulation=runtime.simulation_spec,
            context=dict(params),
            data=data,
        )
    except Exception as exc:
        raise StateBuilderError(
            "InitializerSpec.build_dict(...) failed while building the initial state."
        ) from exc

    return normalize_state_dict(
        runtime=runtime,
        state=raw_state,
        options=options,
    )


def build_initial_state_flat(
    *,
    runtime: RuntimeModel,
    params: ParameterContext,
    data: Any | None = None,
    options: StateBuilderOptions | None = None,
) -> FlatState:
    """
    Build the initial state as a flat vector.

    Even though InitializerSpec exposes build_flat(...), this uses StateLayout
    to flatten so that runtime_model.py remains the single source of truth for
    flat-state ordering.
    """
    options = options or StateBuilderOptions()

    state_dict = build_initial_state_dict(
        runtime=runtime,
        params=params,
        data=data,
        options=options,
    )

    flat_state = runtime.state_layout.flatten(
        state_dict,
        allow_broadcast=False,
    )

    validate_flat_state(
        runtime=runtime,
        flat_state=flat_state,
        options=options,
    )

    return flat_state


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

    _validate_static_values_if_requested(
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

    _validate_static_values_if_requested(
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


def validate_initializer_context(
    *,
    runtime: RuntimeModel,
    params: ParameterContext,
    data: Any | None = None,
) -> None:
    """
    Validate that InitializerSpec has the parameter and data values it needs.

    compile_model.py should already catch most of this statically. This remains
    useful as a runtime guard, especially when data is supplied dynamically.
    """
    initializer = runtime.initializer_spec

    parameter_deps = _dependency_set(
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

    data_deps = _dependency_set(
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

    available_data = _available_data_names(data)
    missing_data = sorted(data_deps - available_data)

    if missing_data:
        raise StateBuilderError(
            "Initializer depends on data series that are not available: "
            f"{missing_data}. Available data series are: {sorted(available_data)}."
        )


def _dependency_set(
    obj: Any,
    attr_name: str,
) -> set[str]:
    attr = getattr(obj, attr_name, None)

    if attr is None:
        return set()

    if callable(attr):
        value = attr()
    else:
        value = attr

    if value is None:
        return set()

    return {str(item) for item in value}


def _available_data_names(data: Any) -> set[str]:
    for attr_name in (
        "observation_names",
        "observed_series_names",
        "data_names",
    ):
        value = getattr(data, attr_name, None)

        if value is None:
            continue

        if callable(value):
            value = value()

        return {str(name) for name in value}

    if isinstance(data, Mapping):
        observations = data.get("observations")

        if isinstance(observations, Mapping):
            return {str(name) for name in observations}

        return {str(name) for name in data}

    observations = getattr(data, "observations", None)

    if observations is None:
        return set()

    if isinstance(observations, Mapping):
        return {str(name) for name in observations}

    names: set[str] = set()

    for observation in observations:
        name = getattr(observation, "name", None)

        if name is not None:
            names.add(str(name))

    return names


def _validate_static_values_if_requested(
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


__all__ = [
    "StateDict",
    "FlatState",
    "ParameterContext",
    "StateBuilderError",
    "StateBuilderOptions",
    "build_initial_state",
    "build_initial_state_dict",
    "build_initial_state_flat",
    "normalize_state_dict",
    "validate_state_dict",
    "validate_flat_state",
    "flatten_state_dict",
    "unflatten_state",
    "state_view",
    "replace_state_view",
    "validate_initializer_context",
]
