from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dynode.runtime.layout.runtime_model import RuntimeModel

from .layouts import (
    compile_parameter_layout,
    compile_state_layout,
    compile_transmission,
)
from .metadata import compile_metadata
from .options import CompileOptions
from .spec_hooks import run_spec_validation_hooks, validate_required_model_shape
from .validators import validate_runtime_model


def compile_model(
    spec: Any,
    *,
    options: CompileOptions | None = None,
) -> RuntimeModel:
    """
    Compile a validated ModelSpec into a RuntimeModel.

    This function is the main bridge between the declarative Pydantic spec layer
    and the runtime execution layer.

    Parameters
    ----------
    spec:
        A ModelSpec-like object with:
        - simulation
        - parameters
        - solver
        - transmission
        - optional data

    options:
        Static compilation options.

    Returns
    -------
    RuntimeModel
        Compiled runtime model containing state layout, parameter layout, and
        transmission layout.

    Notes
    -----
    This function should not:
    - call numpyro.sample
    - resolve deterministic parameters numerically
    - build initial state y0
    - call diffrax.diffeqsolve
    """
    options = options or CompileOptions()

    validate_required_model_shape(spec)

    if options.run_spec_validation_hooks:
        run_spec_validation_hooks(spec, options=options)

    state_layout = compile_state_layout(spec.simulation)

    parameter_layout = compile_parameter_layout(
        spec.parameters,
        options=options,
    )

    transmission = compile_transmission(
        spec.transmission,
        simulation=spec.simulation,
    )

    runtime = RuntimeModel(
        spec=spec,
        state_layout=state_layout,
        parameter_layout=parameter_layout,
        transmission=transmission,
        metadata=compile_metadata(spec, options),
    )

    validate_runtime_model(runtime, options=options)

    return runtime


def compile_model_from_dict(
    data: Mapping[str, Any],
    model_spec_type: type[Any],
    *,
    options: CompileOptions | None = None,
) -> RuntimeModel:
    """
    Convenience helper for loading a RuntimeModel from a dict.

    Example
    -------
    raw = yaml.safe_load(open("model.yaml"))
    runtime = compile_model_from_dict(raw, ModelSpec)
    """
    if not hasattr(model_spec_type, "model_validate"):
        raise TypeError(
            "model_spec_type must be a Pydantic v2 model class exposing "
            "model_validate(...)."
        )

    spec = model_spec_type.model_validate(data)

    return compile_model(
        spec,
        options=options,
    )