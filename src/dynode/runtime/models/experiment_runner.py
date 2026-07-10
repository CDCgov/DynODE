from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dynode.runtime.context.experiment_runtime import (
    ExperimentRuntime,
    ModelInstanceRuntime,
)
from dynode.runtime.context.runtime_context import RuntimeContext
from dynode.runtime.execution.ode_options import OdeSolverOptions
from dynode.runtime.execution.ode_solver import solve_ode
from dynode.runtime.execution.parameter_options import ParameterSamplingOptions
from dynode.runtime.execution.parameter_sampling import sample_parameter_layout
from dynode.runtime.execution.state_builder import build_initial_state

from .experiment_data import (
    build_initial_instance_context,
    build_instance_data_bundle,
    build_local_context,
    select_instance_data,
)
from .types import ObserveFn, RHSFn, StateTransformFn


def run_experiment_trace(
    *,
    runtime: ExperimentRuntime,
    rhs_fn: RHSFn,
    observe_fn: ObserveFn,
    initial_state_transform: StateTransformFn | None,
    parameter_sampling_options: ParameterSamplingOptions,
    ode_solver_options: OdeSolverOptions,
    initial_state_flat: bool,
    return_outputs: bool,
    data: Mapping[str, Any] | None = None,
) -> Any:
    shared_context = sample_parameter_layout(
        parameter_layout=runtime.shared_parameter_layout,
        data=data,
        scope=None,
        options=parameter_sampling_options,
    )

    outputs: dict[str, Any] = {
        "shared_params": shared_context,
        "instances": {},
    }

    for instance in runtime.instances:
        instance_output = run_experiment_instance(
            instance=instance,
            shared_context=shared_context,
            data=data,
            rhs_fn=rhs_fn,
            observe_fn=observe_fn,
            initial_state_transform=initial_state_transform,
            parameter_sampling_options=parameter_sampling_options,
            ode_solver_options=ode_solver_options,
            initial_state_flat=initial_state_flat,
            return_outputs=return_outputs,
        )

        if return_outputs:
            outputs["instances"][instance.key] = instance_output

    return outputs if return_outputs else None


def run_experiment_instance(
    *,
    instance: ModelInstanceRuntime,
    shared_context: Mapping[str, Any],
    rhs_fn: RHSFn,
    observe_fn: ObserveFn,
    initial_state_transform: StateTransformFn | None,
    parameter_sampling_options: ParameterSamplingOptions,
    ode_solver_options: OdeSolverOptions,
    initial_state_flat: bool,
    data: Mapping[str, Any] | None = None,
    return_outputs: bool = False,
) -> Any:
    selected_data = select_instance_data(instance, data)
    static_context = dict(instance.static_context)
    data_bundle = build_instance_data_bundle(
        instance=instance,
        selected_data=selected_data,
    )

    initial_context = build_initial_instance_context(
        shared_context=shared_context,
        static_context=static_context,
    )

    full_context = sample_parameter_layout(
        parameter_layout=instance.parameter_layout,
        data=data_bundle,
        initial_context=initial_context,
        scope=instance.key,
        options=parameter_sampling_options,
    )

    local_context = build_local_context(
        full_context=full_context,
        shared_context=shared_context,
        instance=instance,
    )

    runtime_context = RuntimeContext(
        runtime=instance.runtime,
        params=full_context,
        shared_params=shared_context,
        data=data_bundle,
        scope=instance.key,
        static=static_context,
    )

    y0 = build_initial_state(
        runtime=instance.runtime,
        params=full_context,
        data=data_bundle,
        flat=initial_state_flat,
    )

    if initial_state_transform is not None:
        y0 = initial_state_transform(
            y0=y0,
            context=runtime_context,
            runtime=instance.runtime,
            params=full_context,
            data=data_bundle,
        )

    solution = solve_ode(
        runtime=instance.runtime,
        rhs_fn=rhs_fn,
        y0=y0,
        params=full_context,
        data=data_bundle,
        t0=instance.t0,
        t1=instance.t1,
        options=ode_solver_options,
    )

    observe_result = observe_fn(
        solution=solution,
        params=full_context,
        data=data_bundle,
        runtime=instance.runtime,
        context=runtime_context,
    )

    if not return_outputs:
        return observe_result

    return {
        "params": full_context,
        "local_params": local_context,
        "y0": y0,
        "solution": solution,
        "observe_result": observe_result,
        "data": data_bundle,
        "runtime": instance.runtime,
    }
