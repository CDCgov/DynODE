from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable

from dynode.runtime.layout.runtime_model import RuntimeModel


def default_parameter_sampler(
    *,
    runtime: RuntimeModel,
    data: Any | None = None,
) -> Mapping[str, Any]:
    """
    Default bridge to the parameter-sampling execution module.
    """
    from dynode.runtime.execution.parameter_sampling import sample_parameters

    return sample_parameters(runtime=runtime, data=data)


def default_initial_state_builder(
    *,
    runtime: RuntimeModel,
    params: Mapping[str, Any],
    data: Any | None = None,
    flat: bool = True,
) -> Any:
    """
    Default bridge to the initial-state execution module.
    """
    from dynode.runtime.execution.state_builder import build_initial_state

    return build_initial_state(
        runtime=runtime,
        params=params,
        data=data,
        flat=flat,
    )


def default_ode_solver(
    *,
    runtime: RuntimeModel,
    rhs_fn: Callable[..., Any],
    y0: Any,
    params: Mapping[str, Any],
    data: Any | None = None,
) -> Any:
    """
    Default bridge to the ODE-solver execution module.
    """
    from dynode.runtime.execution.ode_solver import solve_ode

    return solve_ode(
        runtime=runtime,
        rhs_fn=rhs_fn,
        y0=y0,
        params=params,
        data=data,
    )
