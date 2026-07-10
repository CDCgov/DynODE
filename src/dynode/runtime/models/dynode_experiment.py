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
from dynode.runtime.execution.ode_options import OdeSolverOptions
from dynode.runtime.execution.parameter_options import ParameterSamplingOptions

from .callable_validation import require_callable, require_optional_callable
from .experiment_runner import run_experiment_instance, run_experiment_trace
from .types import ObserveFn, RHSFn, StateTransformFn


class DynodeExperiment(BaseModel):
    """
    Executable wrapper for ExperimentSpec.

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
    ode_solver_options: OdeSolverOptions = Field(default_factory=OdeSolverOptions)
    initial_state_flat: bool = True
    cache_runtime: bool = True
    return_outputs: bool = False

    _runtime: ExperimentRuntime | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def validate_callables(self) -> Self:
        require_callable(self.rhs_fn, "rhs_fn")
        require_callable(self.observe_fn, "observe_fn")
        require_optional_callable(
            self.initial_state_transform,
            "initial_state_transform",
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
        self,
        *,
        return_outputs: bool | None = None,
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
        final_return_outputs = (
            self.return_outputs if return_outputs is None else return_outputs
        )

        return run_experiment_trace(
            runtime=runtime,
            rhs_fn=self.rhs_fn,
            observe_fn=self.observe_fn,
            initial_state_transform=self.initial_state_transform,
            parameter_sampling_options=self.parameter_sampling_options,
            ode_solver_options=self.ode_solver_options,
            initial_state_flat=self.initial_state_flat,
            return_outputs=final_return_outputs,
            data=data,
        )

    def run_instance(
        self,
        *,
        instance: ModelInstanceRuntime,
        shared_context: Mapping[str, Any],
        data: Mapping[str, Any] | None = None,
        return_outputs: bool = False,
    ) -> Any:
        return run_experiment_instance(
            instance=instance,
            shared_context=shared_context,
            rhs_fn=self.rhs_fn,
            observe_fn=self.observe_fn,
            initial_state_transform=self.initial_state_transform,
            parameter_sampling_options=self.parameter_sampling_options,
            ode_solver_options=self.ode_solver_options,
            initial_state_flat=self.initial_state_flat,
            data=data,
            return_outputs=return_outputs,
        )
