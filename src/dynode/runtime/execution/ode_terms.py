from __future__ import annotations

from collections.abc import Callable
from typing import Any

import diffrax as dfx

from dynode.runtime.layout.runtime_model import RuntimeModel

from .ode_options import OdeSolverOptions
from .ode_validation import normalize_rhs_output
from .rhs_adapters import make_rhs_adapter
from .types import RHSFn


def make_ode_term(
    *,
    runtime: RuntimeModel,
    rhs_fn: RHSFn,
    data: Any | None = None,
    options: OdeSolverOptions | None = None,
) -> dfx.ODETerm:
    """
    Build a Diffrax ODETerm from the user-provided RHS function.
    """
    return dfx.ODETerm(
        make_vector_field(
            runtime=runtime,
            rhs_fn=rhs_fn,
            data=data,
            options=options,
        )
    )


def make_vector_field(
    *,
    runtime: RuntimeModel,
    rhs_fn: RHSFn,
    data: Any | None = None,
    options: OdeSolverOptions | None = None,
) -> Callable[[Any, Any, Any], Any]:
    """
    Wrap rhs_fn into Diffrax's expected vector field signature:

        vector_field(t, y, args) -> dy_dt

    Here:
    - y is the flat ODE state vector unless rhs_state_format='dict'
    - args is the parameter context
    - runtime and data are captured by closure
    """
    options = options or OdeSolverOptions()

    rhs_adapter = make_rhs_adapter(
        rhs_fn=rhs_fn,
        call_style=options.rhs_call_style,
    )

    def vector_field(t: Any, y: Any, args: Any) -> Any:
        params = args

        if options.rhs_state_format == "dict":
            rhs_state = runtime.state_layout.unflatten(y)
        else:
            rhs_state = y

        rhs_value = rhs_adapter(
            t=t,
            y=rhs_state,
            params=params,
            runtime=runtime,
            data=data,
            extra=options.rhs_extra,
        )

        return normalize_rhs_output(
            runtime=runtime,
            rhs_value=rhs_value,
            validate_shape=options.validate_rhs_output_shape,
        )

    return vector_field
