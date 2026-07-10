from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from dynode.runtime.layout.runtime_model import RuntimeModel


def sample_model_parameters(
    *,
    parameter_sampler: Callable[..., Mapping[str, Any]],
    runtime: RuntimeModel,
    data: Any | None = None,
) -> Mapping[str, Any]:
    return parameter_sampler(runtime=runtime, data=data)


def build_model_initial_state(
    *,
    initial_state_builder: Callable[..., Any],
    runtime: RuntimeModel,
    params: Mapping[str, Any],
    data: Any | None = None,
    flat: bool = True,
) -> Any:
    return initial_state_builder(
        runtime=runtime,
        params=params,
        data=data,
        flat=flat,
    )


def solve_model_ode(
    *,
    ode_solver: Callable[..., Any],
    runtime: RuntimeModel,
    rhs_fn: Callable[..., Any],
    y0: Any,
    params: Mapping[str, Any],
    data: Any | None = None,
) -> Any:
    return ode_solver(
        runtime=runtime,
        rhs_fn=rhs_fn,
        y0=y0,
        params=params,
        data=data,
    )


def observe_model_solution(
    *,
    observe_fn: Callable[..., Any],
    runtime: RuntimeModel,
    solution: Any,
    params: Mapping[str, Any],
    data: Any | None = None,
) -> Any:
    return observe_fn(
        solution=solution,
        params=params,
        data=data,
        runtime=runtime,
    )


def run_single_model_trace(
    *,
    runtime: RuntimeModel,
    rhs_fn: Callable[..., Any],
    observe_fn: Callable[..., Any],
    parameter_sampler: Callable[..., Mapping[str, Any]],
    initial_state_builder: Callable[..., Any],
    ode_solver: Callable[..., Any],
    initial_state_flat: bool,
    return_outputs: bool,
    data: Any | None = None,
) -> Any:
    params = sample_model_parameters(
        parameter_sampler=parameter_sampler,
        runtime=runtime,
        data=data,
    )

    y0 = build_model_initial_state(
        initial_state_builder=initial_state_builder,
        runtime=runtime,
        params=params,
        data=data,
        flat=initial_state_flat,
    )

    solution = solve_model_ode(
        ode_solver=ode_solver,
        runtime=runtime,
        rhs_fn=rhs_fn,
        y0=y0,
        params=params,
        data=data,
    )

    observe_result = observe_model_solution(
        observe_fn=observe_fn,
        runtime=runtime,
        solution=solution,
        params=params,
        data=data,
    )

    if not return_outputs:
        return observe_result

    return {
        "runtime": runtime,
        "params": params,
        "y0": y0,
        "solution": solution,
        "observe_result": observe_result,
    }
