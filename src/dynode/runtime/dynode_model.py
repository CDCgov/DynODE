from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    model_validator,
)

# Adjust this import path if your ModelSpec lives somewhere else.
from specs.model_spec import ModelSpec
from typing_extensions import Self

from .compile_model import CompileOptions, compile_model
from .runtime_model import RuntimeModel


def _default_parameter_sampler(
    *,
    runtime: RuntimeModel,
    data: Any | None = None,
) -> Mapping[str, Any]:
    """
    Default bridge to parameter_sampling.py.

    Imported lazily so dynode_model.py can be imported before all runtime
    modules are fully implemented.
    """
    from .parameter_sampling import sample_parameters

    return sample_parameters(
        runtime=runtime,
        data=data,
    )


def _default_initial_state_builder(
    *,
    runtime: RuntimeModel,
    params: Mapping[str, Any],
    data: Any | None = None,
    flat: bool = True,
) -> Any:
    """
    Default bridge to state_builder.py.
    """
    from .state_builder import build_initial_state

    return build_initial_state(
        runtime=runtime,
        params=params,
        data=data,
        flat=flat,
    )


def _default_ode_solver(
    *,
    runtime: RuntimeModel,
    rhs_fn: Callable[..., Any],
    y0: Any,
    params: Mapping[str, Any],
    data: Any | None = None,
) -> Any:
    """
    Default bridge to ode_solver.py.
    """
    from .ode_solver import solve_ode

    return solve_ode(
        runtime=runtime,
        rhs_fn=rhs_fn,
        y0=y0,
        params=params,
        data=data,
    )


class DynodeModel(BaseModel):
    """
    Executable wrapper around a declarative ModelSpec.

    DynodeModel is the orchestration layer. It connects:

        ModelSpec
            ↓
        compile_model(...)
            ↓
        RuntimeModel
            ↓
        sample_parameters(...)
            ↓
        build_initial_state(...)
            ↓
        solve_ode(...)
            ↓
        observe_fn(...)

    This class should not itself contain epidemiological model equations,
    NumPyro priors, Diffrax solver implementation details, or state-layout
    compilation logic. Those belong in the injected functions/modules.

    Expected function signatures
    ----------------------------
    rhs_fn:
        Called by ode_solver.py. Typically compatible with Diffrax:

            rhs_fn(t, y, args)

        where args will usually contain params, runtime, and data.

    observe_fn:
        Called after the ODE solve:

            observe_fn(
                solution=solution,
                params=params,
                data=data,
                runtime=runtime,
            )

        This function is where NumPyro likelihood statements should usually
        live.
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
        default=_default_parameter_sampler,
        exclude=True,
        description="Function that samples priors and resolves deterministic parameters.",
    )

    initial_state_builder: Callable[..., Any] = Field(
        default=_default_initial_state_builder,
        exclude=True,
        description="Function that builds the initial ODE state.",
    )

    ode_solver: Callable[..., Any] = Field(
        default=_default_ode_solver,
        exclude=True,
        description="Function that solves the ODE.",
    )

    _runtime: RuntimeModel | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def validate_model(self) -> Self:
        if not callable(self.rhs_fn):
            raise TypeError("rhs_fn must be callable.")

        if not callable(self.observe_fn):
            raise TypeError("observe_fn must be callable.")

        if not callable(self.parameter_sampler):
            raise TypeError("parameter_sampler must be callable.")

        if not callable(self.initial_state_builder):
            raise TypeError("initial_state_builder must be callable.")

        if not callable(self.ode_solver):
            raise TypeError("ode_solver must be callable.")

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

    def compile(
        self,
        *,
        force: bool = False,
    ) -> RuntimeModel:
        """
        Compile the declarative ModelSpec into a RuntimeModel.

        Parameters
        ----------
        force:
            If True, rebuild the RuntimeModel even if a cached version exists.
        """
        if self.cache_runtime and self._runtime is not None and not force:
            return self._runtime

        runtime = compile_model(
            self.spec,
            options=self.compile_options,
        )

        if self.cache_runtime:
            self._runtime = runtime

        return runtime

    def clear_runtime_cache(self) -> None:
        """
        Clear the cached RuntimeModel.

        Useful if you mutate specs during development. In production, specs
        should generally be frozen/immutable.
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
        """
        Sample priors and resolve deterministic parameters.

        This delegates to parameter_sampling.py by default.
        """
        runtime = runtime or self.compile()

        return self.parameter_sampler(
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
        """
        Build initial ODE state y0.

        This delegates to state_builder.py by default.
        """
        runtime = runtime or self.compile()

        if flat is None:
            flat = self.initial_state_flat

        return self.initial_state_builder(
            runtime=runtime,
            params=params,
            data=data,
            flat=flat,
        )

    def solve(
        self,
        *,
        runtime: RuntimeModel | None = None,
        y0: Any,
        params: Mapping[str, Any],
        data: Any | None = None,
    ) -> Any:
        """
        Solve the ODE.

        This delegates to ode_solver.py by default.
        """
        runtime = runtime or self.compile()

        return self.ode_solver(
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
        """
        Apply the observation model / likelihood.

        observe_fn should usually contain NumPyro likelihood statements.
        """
        runtime = runtime or self.compile()

        return self.observe_fn(
            solution=solution,
            params=params,
            data=data,
            runtime=runtime,
        )

    def run_once(
        self,
        *,
        data: Any | None = None,
        runtime: RuntimeModel | None = None,
        return_outputs: bool | None = None,
    ) -> Any:
        """
        Execute one model trace.

        Inside NumPyro, this method should be called from the generated
        numpyro_model function. It samples parameters, builds y0, solves the
        ODE, and applies observe_fn.
        """
        runtime = runtime or self.compile()

        params = self.sample_parameters(
            runtime=runtime,
            data=data,
        )

        y0 = self.build_initial_state(
            runtime=runtime,
            params=params,
            data=data,
            flat=self.initial_state_flat,
        )

        solution = self.solve(
            runtime=runtime,
            y0=y0,
            params=params,
            data=data,
        )

        observe_result = self.observe(
            runtime=runtime,
            solution=solution,
            params=params,
            data=data,
        )

        if return_outputs is None:
            return_outputs = self.return_outputs

        if return_outputs:
            return {
                "runtime": runtime,
                "params": params,
                "y0": y0,
                "solution": solution,
                "observe_result": observe_result,
            }

        return observe_result

    def make_numpyro_model(
        self,
        *,
        compile_now: bool = True,
        return_outputs: bool | None = None,
    ) -> Callable[..., Any]:
        """
        Build a NumPyro-compatible model function.

        Returns
        -------
        Callable
            A function suitable for NumPyro inference APIs.

        Example
        -------
        numpyro_model = dynode_model.make_numpyro_model()

        mcmc = numpyro.infer.MCMC(...)
        mcmc.run(rng_key, data=observed_data)

        Notes
        -----
        The returned function has signature:

            numpyro_model(data=None)

        You can pass any data object expected by your DataSpec, initializer,
        deterministic expressions, or observe_fn.
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

    def __call__(
        self,
        data: Any | None = None,
    ) -> Any:
        """
        Execute one model trace directly.

        This is mostly a convenience for testing or for using DynodeModel itself
        as a NumPyro model callable.
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
