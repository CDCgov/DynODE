from __future__ import annotations

from typing import Any

from dynode.runtime.layout.runtime_model import RuntimeModel

from .errors import StateBuilderError
from .state_options import StateBuilderOptions
from .state_validation import (
    normalize_state_dict,
    validate_flat_state,
    validate_initializer_context,
)
from .types import FlatState, ParameterMapping, StateDict


def build_initial_state(
    *,
    runtime: RuntimeModel,
    params: ParameterMapping,
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
    params: ParameterMapping,
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
    params: ParameterMapping,
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
