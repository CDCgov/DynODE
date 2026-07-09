from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator
from typing_extensions import Self

from dynode.experiment.experiment_spec import ExperimentSpec
from dynode.runtime.compile.compile_experiment import compile_experiment
from dynode.runtime.context.experiment_runtime import (
    ExperimentRuntime,
    ModelInstanceRuntime,
)
from dynode.runtime.context.runtime_context import RuntimeContext
from dynode.runtime.execution.ode_solver import OdeSolverOptions, solve_ode
from dynode.runtime.execution.parameter_sampling import (
    ParameterSamplingOptions,
    sample_parameter_layout,
)
from dynode.runtime.execution.state_builder import build_initial_state

StateTransformFn = Callable[..., Any]
ObserveFn = Callable[..., Any]
RHSFn = Callable[..., Any]


class DynodeExperiment(BaseModel):
    """Executable wrapper for ExperimentSpec.

    This is the multi-instance complement to DynodeModel. It samples shared
    parameters once, samples instance-local parameters under a scope, then runs
    the state builder, ODE solver, and observation function for each instance.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    spec: ExperimentSpec
    rhs_fn: RHSFn
    observe_fn: ObserveFn
    initial_state_transform: StateTransformFn | None = None

    parameter_sampling_options: ParameterSamplingOptions = Field(
        default_factory=ParameterSamplingOptions
    )
    ode_solver_options: OdeSolverOptions = Field(
        default_factory=OdeSolverOptions
    )
    initial_state_flat: bool = True
    cache_runtime: bool = True
    return_outputs: bool = False

    _runtime: ExperimentRuntime | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def validate_callables(self) -> Self:
        if not callable(self.rhs_fn):
            raise TypeError("rhs_fn must be callable.")
        if not callable(self.observe_fn):
            raise TypeError("observe_fn must be callable.")
        if self.initial_state_transform is not None and not callable(
            self.initial_state_transform
        ):
            raise TypeError(
                "initial_state_transform must be callable if provided."
            )
        return self

    def compile(self, *, force: bool = False) -> ExperimentRuntime:
        if self.cache_runtime and self._runtime is not None and not force:
            return self._runtime
        runtime = compile_experiment(self.spec)
        if self.cache_runtime:
            self._runtime = runtime
        return runtime

    @property
    def runtime(self) -> ExperimentRuntime:
        return self.compile()

    def make_numpyro_model(
        self, *, return_outputs: bool | None = None
    ) -> Callable[..., Any]:
        runtime = self.compile()
        final_return_outputs = (
            self.return_outputs if return_outputs is None else return_outputs
        )

        def numpyro_model(data: Mapping[str, Any] | None = None) -> Any:
            return self.run_once(
                data=data,
                runtime=runtime,
                return_outputs=final_return_outputs,
            )

        return numpyro_model

    def run_once(
        self,
        *,
        data: Mapping[str, Any] | None = None,
        runtime: ExperimentRuntime | None = None,
        return_outputs: bool | None = None,
    ) -> Any:
        runtime = runtime or self.compile()
        return_outputs = (
            self.return_outputs if return_outputs is None else return_outputs
        )

        shared_context = sample_parameter_layout(
            parameter_layout=runtime.shared_parameter_layout,
            data=data,
            scope=None,
            options=self.parameter_sampling_options,
        )

        outputs: dict[str, Any] = {
            "shared_params": shared_context,
            "instances": {},
        }

        for instance in runtime.instances:
            instance_output = self.run_instance(
                instance=instance,
                shared_context=shared_context,
                data=data,
                return_outputs=return_outputs,
            )
            if return_outputs:
                outputs["instances"][instance.key] = instance_output

        return outputs if return_outputs else None

    def run_instance(
        self,
        *,
        instance: ModelInstanceRuntime,
        shared_context: Mapping[str, Any],
        data: Mapping[str, Any] | None = None,
        return_outputs: bool = False,
    ) -> Any:
        selected_data = _select_instance_data(instance, data)
        static_context = dict(instance.static_context)
        if isinstance(selected_data, Mapping):
            data_bundle = dict(static_context)
            data_bundle.update(dict(selected_data))
        elif selected_data is None:
            data_bundle = dict(static_context)
        else:
            data_bundle = selected_data

        initial_context = dict(shared_context)
        initial_context.update(static_context.get("parameter_context", {}))

        full_context = sample_parameter_layout(
            parameter_layout=instance.parameter_layout,
            data=data_bundle,
            initial_context=initial_context,
            scope=instance.key,
            options=self.parameter_sampling_options,
        )

        local_context = {
            key: value
            for key, value in full_context.items()
            if key not in shared_context
            or key in instance.parameter_layout.resolved_name_set
        }

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
            flat=self.initial_state_flat,
        )

        if self.initial_state_transform is not None:
            y0 = self.initial_state_transform(
                y0=y0,
                context=runtime_context,
                runtime=instance.runtime,
                params=full_context,
                data=data_bundle,
            )

        solution = solve_ode(
            runtime=instance.runtime,
            rhs_fn=self.rhs_fn,
            y0=y0,
            params=full_context,
            data=data_bundle,
            t0=instance.t0,
            t1=instance.t1,
            options=self.ode_solver_options,
        )

        observe_result = self.observe_fn(
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


def _select_instance_data(
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


__all__ = ["DynodeExperiment", "StateTransformFn", "ObserveFn", "RHSFn"]
