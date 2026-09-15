from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    model_validator,
)
from typing_extensions import Self

from dynode.runtime.compile.compile_model import compile_model
from dynode.runtime.compile.options import CompileOptions
from dynode.runtime.layout.runtime_model import RuntimeModel
from dynode.structure.model.model_spec import ModelSpec

from .callable_validation import require_callable
from .model_defaults import (
    default_initial_state_builder,
    default_ode_solver,
    default_parameter_sampler,
)
from .model_runner import (
    build_model_initial_state,
    observe_model_solution,
    run_single_model_trace,
    sample_model_parameters,
    solve_model_ode,
)


class DynodeModel(BaseModel):
    """
    Executable wrapper around a declarative ModelSpec.

    DynodeModel is the single-model orchestration layer. It connects model
    compilation, parameter sampling, initial-state construction, ODE solving,
    and observation/likelihood evaluation without owning those implementations.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        validate_assignment=True,
        populate_by_name=True,
    )

    spec: ModelSpec = Field(
        description="Declarative model specification.",
    )

    rhs_fn: Callable[..., Any] = Field(
        validation_alias=AliasChoices("rhs_fn", "ode"),
        description=(
            "ODE right-hand-side function. The legacy field name 'ode' is "
            "accepted as an alias."
        ),
    )

    observe_fn: Callable[..., Any] = Field(
        description=(
            "Observation/likelihood function. Usually contains NumPyro "
            "observe statements."
        ),
    )

    compile_options: CompileOptions = Field(
        default_factory=CompileOptions,
        description="Options passed to compile_model(...).",
    )

    cache_runtime: bool = Field(
        default=True,
        description=(
            "If True, cache the compiled RuntimeModel after the first compile."
        ),
    )

    initial_state_flat: bool = Field(
        default=True,
        description=(
            "If True, build y0 as a flat state vector. This should usually be "
            "True for Diffrax solves using the compiled StateLayout."
        ),
    )

    return_outputs: bool = Field(
        default=False,
        description=(
            "If True, the generated numpyro_model returns a dictionary with "
            "params, y0, solution, and observe_result. If False, it returns "
            "only observe_result."
        ),
    )

    parameter_sampler: Callable[..., Mapping[str, Any]] = Field(
        default=default_parameter_sampler,
        exclude=True,
        description="Function that samples priors and resolves deterministic parameters.",
    )

    initial_state_builder: Callable[..., Any] = Field(
        default=default_initial_state_builder,
        exclude=True,
        description="Function that builds the initial ODE state.",
    )

    ode_solver: Callable[..., Any] = Field(
        default=default_ode_solver,
        exclude=True,
        description="Function that solves the ODE.",
    )

    _runtime: RuntimeModel | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def validate_model(self) -> Self:
        require_callable(self.rhs_fn, "rhs_fn")
        require_callable(self.observe_fn, "observe_fn")
        require_callable(self.parameter_sampler, "parameter_sampler")
        require_callable(self.initial_state_builder, "initial_state_builder")
        require_callable(self.ode_solver, "ode_solver")
        return self

    @property
    def name(self) -> str:
        return str(getattr(self.spec, "name", self.__class__.__name__))

    @property
    def version(self) -> str | None:
        return getattr(self.spec, "version", None)

    @property
    def ode(self) -> Callable[..., Any]:
        """
        Backward-compatible alias for rhs_fn.
        """
        return self.rhs_fn

    def compile(self, *, force: bool = False) -> RuntimeModel:
        """
        Compile the declarative ModelSpec into a RuntimeModel.
        """
        if self.cache_runtime and self._runtime is not None and not force:
            return self._runtime

        runtime = compile_model(self.spec, options=self.compile_options)

        if self.cache_runtime:
            self._runtime = runtime

        return runtime

    def clear_runtime_cache(self) -> None:
        """
        Clear the cached RuntimeModel.
        """
        self._runtime = None

    @property
    def runtime(self) -> RuntimeModel:
        """
        Lazily compiled RuntimeModel.
        """
        return self.compile()

    def sample_parameters(
        self,
        *,
        runtime: RuntimeModel | None = None,
        data: Any | None = None,
    ) -> Mapping[str, Any]:
        runtime = runtime or self.compile()
        return sample_model_parameters(
            parameter_sampler=self.parameter_sampler,
            runtime=runtime,
            data=data,
        )

    def build_initial_state(
        self,
        *,
        runtime: RuntimeModel | None = None,
        params: Mapping[str, Any],
        data: Any | None = None,
        flat: bool | None = None,
    ) -> Any:
        runtime = runtime or self.compile()
        final_flat = self.initial_state_flat if flat is None else flat
        return build_model_initial_state(
            initial_state_builder=self.initial_state_builder,
            runtime=runtime,
            params=params,
            data=data,
            flat=final_flat,
        )

    def solve(
        self,
        *,
        runtime: RuntimeModel | None = None,
        y0: Any,
        params: Mapping[str, Any],
        data: Any | None = None,
    ) -> Any:
        runtime = runtime or self.compile()
        return solve_model_ode(
            ode_solver=self.ode_solver,
            runtime=runtime,
            rhs_fn=self.rhs_fn,
            y0=y0,
            params=params,
            data=data,
        )

    def observe(
        self,
        *,
        runtime: RuntimeModel | None = None,
        solution: Any,
        params: Mapping[str, Any],
        data: Any | None = None,
    ) -> Any:
        runtime = runtime or self.compile()
        return observe_model_solution(
            observe_fn=self.observe_fn,
            runtime=runtime,
            solution=solution,
            params=params,
            data=data,
        )

    def run_once(
        self,
        *,
        data: Any | None = None,
        runtime: RuntimeModel | None = None,
        return_outputs: bool | None = None,
    ) -> Any:
        runtime = runtime or self.compile()
        final_return_outputs = (
            self.return_outputs if return_outputs is None else return_outputs
        )
        return run_single_model_trace(
            runtime=runtime,
            rhs_fn=self.rhs_fn,
            observe_fn=self.observe_fn,
            parameter_sampler=self.parameter_sampler,
            initial_state_builder=self.initial_state_builder,
            ode_solver=self.ode_solver,
            initial_state_flat=self.initial_state_flat,
            return_outputs=final_return_outputs,
            data=data,
        )

    def make_numpyro_model(
        self,
        *,
        compile_now: bool = True,
        return_outputs: bool | None = None,
    ) -> Callable[..., Any]:
        """
        Build a NumPyro-compatible model function.
        """
        compiled_runtime = self.compile() if compile_now else None

        def numpyro_model(data: Any | None = None) -> Any:
            runtime = compiled_runtime or self.compile()
            return self.run_once(
                data=data,
                runtime=runtime,
                return_outputs=return_outputs,
            )

        numpyro_model.__name__ = f"{self.name}_numpyro_model"
        return numpyro_model

    def __call__(self, data: Any | None = None) -> Any:
        """
        Execute one model trace directly.
        """
        return self.run_once(data=data)

    def summary(self) -> dict[str, Any]:
        """
        Return a lightweight summary useful for debugging and logging.
        """
        runtime = self.compile()
        return {
            "name": self.name,
            "version": self.version,
            "runtime": runtime.summary(),
            "cache_runtime": self.cache_runtime,
            "initial_state_flat": self.initial_state_flat,
        }
