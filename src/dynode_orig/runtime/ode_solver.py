from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import diffrax as dfx
import jax.numpy as jnp
import numpy as np

from .runtime_model import RuntimeModel

ParameterContext = Mapping[str, Any]
FlatState = Any
StateDict = Mapping[str, Any]
RHSFn = Callable[..., Any]

RHSStateFormat = Literal["flat", "dict"]
RHSCallStyle = Literal["auto", "standard", "keyword"]


class OdeSolverError(RuntimeError):
    """
    Raised when the ODE solve cannot be configured or completed.
    """


@dataclass(frozen=True, slots=True)
class OdeSolverOptions:
    """
    Runtime options for wrapping and validating the ODE solve.

    This class intentionally does not duplicate SolverSpec. SolverSpec owns:
    - solver method
    - dt0
    - step-size controller
    - save_at
    - max_steps
    - throw
    - jump_ts / step_ts handling
    """

    rhs_state_format: RHSStateFormat = "flat"
    rhs_call_style: RHSCallStyle = "auto"

    validate_y0: bool = True
    validate_rhs_output_shape: bool = True
    validate_solution_shape: bool = True

    y0_dtype: Any | None = None

    rhs_extra: Mapping[str, Any] = field(default_factory=dict)


def solve_ode(
    *,
    runtime: RuntimeModel,
    rhs_fn: RHSFn,
    y0: FlatState | StateDict,
    params: ParameterContext,
    data: Any | None = None,
    t0: float | None = None,
    t1: float | None = None,
    terms: Any | None = None,
    extra_diffeqsolve_kwargs: Mapping[str, Any] | None = None,
    options: OdeSolverOptions | None = None,
) -> dfx.Solution:
    """
    Solve the ODE for one parameter context and initial state.

    Parameters
    ----------
    runtime:
        Compiled RuntimeModel.

    rhs_fn:
        ODE right-hand-side function.

    y0:
        Initial state. May be either a flat vector or a compartment dictionary.

    params:
        Full parameter context from parameter_sampling.py.

    data:
        Optional data object passed through to rhs_fn.

    t0, t1:
        Optional solve start/end times. If omitted, the solver tries to infer
        them from data, runtime.data_spec, or SolverSpec.save_at.ts.

    terms:
        Optional prebuilt Diffrax term. Usually None.

    extra_diffeqsolve_kwargs:
        Optional extra Diffrax kwargs that are not owned by SolverSpec and not
        owned by solve_ode. Examples might include advanced Diffrax arguments
        such as an adjoint or event configuration.

    options:
        Runtime solve wrapper options.
    """
    options = options or OdeSolverOptions()

    if not callable(rhs_fn):
        raise OdeSolverError("rhs_fn must be callable.")

    flat_y0 = prepare_initial_state_for_solve(
        runtime=runtime,
        y0=y0,
        options=options,
    )

    resolved_t0, resolved_t1 = resolve_time_span(
        runtime=runtime,
        data=data,
        t0=t0,
        t1=t1,
    )

    diffeqsolve_kwargs = build_diffeqsolve_kwargs(
        runtime=runtime,
        extra_kwargs=extra_diffeqsolve_kwargs,
    )

    if terms is None:
        terms = make_ode_term(
            runtime=runtime,
            rhs_fn=rhs_fn,
            data=data,
            options=options,
        )

    try:
        solution = dfx.diffeqsolve(
            terms=terms,
            t0=resolved_t0,
            t1=resolved_t1,
            y0=flat_y0,
            args=dict(params),
            **diffeqsolve_kwargs,
        )
    except Exception as exc:
        raise OdeSolverError("Diffrax ODE solve failed.") from exc

    if options.validate_solution_shape:
        validate_solution(
            runtime=runtime,
            solution=solution,
            options=options,
        )

    return solution


def build_diffeqsolve_kwargs(
    *,
    runtime: RuntimeModel,
    extra_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build keyword arguments for diffrax.diffeqsolve.

    SolverSpec is the only source of truth for:
    - solver
    - dt0
    - stepsize_controller
    - saveat
    - max_steps
    - throw
    """
    solver_spec = runtime.solver_spec

    diffeqsolve_kwargs = getattr(solver_spec, "diffeqsolve_kwargs", None)

    if not callable(diffeqsolve_kwargs):
        raise OdeSolverError(
            "runtime.solver_spec must expose diffeqsolve_kwargs(). "
            "Do not reconstruct Diffrax solver objects in ode_solver.py."
        )

    kwargs = dict(diffeqsolve_kwargs())

    required = {
        "solver",
        "dt0",
        "stepsize_controller",
        "saveat",
        "max_steps",
        "throw",
    }

    missing = sorted(required - set(kwargs))

    if missing:
        raise OdeSolverError(
            "SolverSpec.diffeqsolve_kwargs() did not provide required keys: "
            f"{missing}."
        )

    if extra_kwargs:
        _validate_extra_diffeqsolve_kwargs(extra_kwargs)
        kwargs.update(dict(extra_kwargs))

    return kwargs


def _validate_extra_diffeqsolve_kwargs(
    extra_kwargs: Mapping[str, Any],
) -> None:
    """
    Prevent extra kwargs from overriding values owned by solve_ode or SolverSpec.
    """
    solve_owned = {
        "terms",
        "t0",
        "t1",
        "y0",
        "args",
    }

    solver_spec_owned = {
        "solver",
        "dt0",
        "stepsize_controller",
        "saveat",
        "max_steps",
        "throw",
    }

    reserved = solve_owned | solver_spec_owned
    conflicts = sorted(reserved & set(extra_kwargs))

    if conflicts:
        raise OdeSolverError(
            "extra_diffeqsolve_kwargs cannot override arguments owned by "
            f"solve_ode or SolverSpec: {conflicts}."
        )


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

    rhs_adapter = _make_rhs_adapter(
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


def prepare_initial_state_for_solve(
    *,
    runtime: RuntimeModel,
    y0: FlatState | StateDict,
    options: OdeSolverOptions | None = None,
) -> FlatState:
    """
    Coerce y0 into the flat vector representation used by Diffrax.

    state_builder.py should usually provide a flat vector already. This function
    is only a boundary check.
    """
    options = options or OdeSolverOptions()

    if isinstance(y0, Mapping):
        flat_y0 = runtime.state_layout.flatten(
            y0,
            allow_broadcast=False,
        )
    else:
        flat_y0 = jnp.asarray(y0)

    if options.y0_dtype is not None:
        flat_y0 = flat_y0.astype(options.y0_dtype)

    if options.validate_y0:
        try:
            runtime.state_layout.validate_flat_state(flat_y0)
        except Exception as exc:
            raise OdeSolverError(
                "Initial state y0 does not match RuntimeModel.state_layout."
            ) from exc

    return flat_y0


def resolve_time_span(
    *,
    runtime: RuntimeModel,
    data: Any | None = None,
    t0: float | None = None,
    t1: float | None = None,
) -> tuple[float, float]:
    """
    Resolve solve start/end times.

    Priority:
    1. Explicit t0/t1 arguments
    2. data.time
    3. runtime.data_spec.time
    4. runtime.solver_spec.save_at.ts
    """
    if t0 is not None and t1 is not None:
        return float(t0), float(t1)

    candidate_times = (
        _extract_time_values(data)
        or _extract_time_values(runtime.data_spec)
        or _extract_time_values_from_solver_spec(runtime.solver_spec)
    )

    if candidate_times is None or len(candidate_times) < 2:
        missing = []

        if t0 is None:
            missing.append("t0")

        if t1 is None:
            missing.append("t1")

        raise OdeSolverError(
            "Could not infer ODE solve time span. Provide explicit "
            f"{missing}, or define data.time / runtime.data_spec.time / "
            "solver.save_at.ts."
        )

    inferred_t0 = float(candidate_times[0])
    inferred_t1 = float(candidate_times[-1])

    return (
        float(t0) if t0 is not None else inferred_t0,
        float(t1) if t1 is not None else inferred_t1,
    )


def _extract_time_values(source: Any | None) -> tuple[float, ...] | None:
    if source is None:
        return None

    if isinstance(source, Mapping):
        if "time" not in source:
            return None

        return _coerce_time_values(source["time"])

    time_obj = getattr(source, "time", None)

    if time_obj is None:
        return None

    return _coerce_time_values(time_obj)


def _coerce_time_values(time_obj: Any) -> tuple[float, ...] | None:
    if time_obj is None:
        return None

    if isinstance(time_obj, Mapping):
        if "values" in time_obj:
            return tuple(float(value) for value in time_obj["values"])

        if "ts" in time_obj:
            return tuple(float(value) for value in time_obj["ts"])

        return None

    values = getattr(time_obj, "values", None)

    if values is not None:
        return tuple(float(value) for value in values)

    as_numpy = getattr(time_obj, "as_numpy", None)

    if callable(as_numpy):
        return tuple(float(value) for value in as_numpy())

    as_jax = getattr(time_obj, "as_jax", None)

    if callable(as_jax):
        return tuple(float(value) for value in np.asarray(as_jax()))

    if isinstance(time_obj, (list, tuple)):
        return tuple(float(value) for value in time_obj)

    return None


def _extract_time_values_from_solver_spec(
    solver_spec: Any,
) -> tuple[float, ...] | None:
    save_at = getattr(solver_spec, "save_at", None)

    if save_at is None:
        return None

    ts = getattr(save_at, "ts", None)

    if ts is None:
        return None

    return tuple(float(value) for value in ts)


def _make_rhs_adapter(
    *,
    rhs_fn: RHSFn,
    call_style: RHSCallStyle,
) -> Callable[..., Any]:
    if call_style == "standard":
        return _make_standard_rhs_adapter(rhs_fn)

    if call_style == "keyword":
        return _make_keyword_rhs_adapter(rhs_fn)

    if call_style != "auto":
        raise OdeSolverError(
            f"Unknown rhs_call_style {call_style!r}. "
            "Expected 'auto', 'standard', or 'keyword'."
        )

    signature = inspect.signature(rhs_fn)
    parameter_names = set(signature.parameters)

    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )

    keyword_names = {
        "params",
        "runtime",
        "data",
        "state_layout",
        "transmission",
        "extra",
    }

    if accepts_kwargs or parameter_names & keyword_names:
        return _make_keyword_rhs_adapter(rhs_fn)

    return _make_standard_rhs_adapter(rhs_fn)


def _make_standard_rhs_adapter(
    rhs_fn: RHSFn,
) -> Callable[..., Any]:
    """
    Adapter for RHS functions of shape:

        rhs_fn(t, y, args)

    where args is a dictionary containing params, runtime, data, and helpers.
    """

    def adapter(
        *,
        t: Any,
        y: Any,
        params: ParameterContext,
        runtime: RuntimeModel,
        data: Any | None,
        extra: Mapping[str, Any],
    ) -> Any:
        rhs_args = {
            "params": params,
            "runtime": runtime,
            "data": data,
            "state_layout": runtime.state_layout,
            "transmission": runtime.transmission,
            "extra": extra,
        }

        return rhs_fn(t, y, rhs_args)

    return adapter


def _make_keyword_rhs_adapter(
    rhs_fn: RHSFn,
) -> Callable[..., Any]:
    """
    Adapter for RHS functions using named arguments.

    Supported examples:
        rhs_fn(t=t, y=y, params=params, runtime=runtime)
        rhs_fn(t, y, params=params, runtime=runtime)
        rhs_fn(t, y, *, params, runtime, data=None)

    Important: do not catch broad exceptions around the RHS call. If the RHS
    raises a real tracing/runtime error, catching it and retrying with a
    different signature obscures the true root cause. Select the call style from
    the function signature before invoking the RHS.
    """
    signature = inspect.signature(rhs_fn)

    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )

    def _can_pass_as_keyword(parameter_name: str) -> bool:
        parameter = signature.parameters.get(parameter_name)
        if parameter is None:
            return False
        return parameter.kind in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }

    positional_parameters = tuple(
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind
        in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        }
    )

    pass_t_y_by_keyword = _can_pass_as_keyword("t") and _can_pass_as_keyword(
        "y"
    )
    pass_t_y_positionally = (
        not pass_t_y_by_keyword and len(positional_parameters) >= 2
    )

    def adapter(
        *,
        t: Any,
        y: Any,
        params: ParameterContext,
        runtime: RuntimeModel,
        data: Any | None,
        extra: Mapping[str, Any],
    ) -> Any:
        kwargs = {
            "t": t,
            "y": y,
            "params": params,
            "runtime": runtime,
            "data": data,
            "state_layout": runtime.state_layout,
            "transmission": runtime.transmission,
            "extra": extra,
        }

        if accepts_kwargs:
            supported_kwargs = kwargs
        else:
            supported_kwargs = {
                key: value
                for key, value in kwargs.items()
                if key in signature.parameters
            }

        if pass_t_y_by_keyword:
            return rhs_fn(**supported_kwargs)

        if pass_t_y_positionally:
            supported_kwargs = dict(supported_kwargs)
            supported_kwargs.pop("t", None)
            supported_kwargs.pop("y", None)
            return rhs_fn(t, y, **supported_kwargs)

        return rhs_fn(**supported_kwargs)

    return adapter


def normalize_rhs_output(
    *,
    runtime: RuntimeModel,
    rhs_value: Any,
    validate_shape: bool = True,
) -> Any:
    """
    Normalize rhs_fn output into a flat dy/dt vector.

    rhs_fn may return either:
    - a flat vector
    - a compartment dictionary
    """
    if isinstance(rhs_value, Mapping):
        flat_rhs = runtime.state_layout.flatten(
            rhs_value,
            allow_broadcast=False,
        )
    else:
        flat_rhs = jnp.asarray(rhs_value)

    if validate_shape:
        expected_shape = (runtime.total_state_size,)
        actual_shape = tuple(flat_rhs.shape)

        if actual_shape != expected_shape:
            raise OdeSolverError(
                "rhs_fn returned an array with the wrong shape. "
                f"Expected {expected_shape}, got {actual_shape}."
            )

    return flat_rhs


def validate_solution(
    *,
    runtime: RuntimeModel,
    solution: dfx.Solution,
    options: OdeSolverOptions | None = None,
) -> None:
    """
    Validate the shape of a Diffrax solution when possible.
    """
    options = options or OdeSolverOptions()

    if not options.validate_solution_shape:
        return

    ys = getattr(solution, "ys", None)

    if ys is None:
        return

    try:
        ys_array = jnp.asarray(ys)
    except Exception:
        # SaveAt(subs=...) or custom output structures may return pytrees.
        return

    if ys_array.ndim == 1:
        final_dim = ys_array.shape[0]
    else:
        final_dim = ys_array.shape[-1]

    if final_dim != runtime.total_state_size:
        raise OdeSolverError(
            "Diffrax solution does not appear to contain flat state vectors. "
            f"Expected final dimension {runtime.total_state_size}, got {final_dim}."
        )


def solution_ys_as_state_dict(
    *,
    runtime: RuntimeModel,
    solution: dfx.Solution,
) -> dict[str, Any]:
    """
    Convert solution.ys from flat vectors into compartment arrays.

    If solution.ys has shape:

        (n_times, total_state_size)

    then each returned compartment has shape:

        (n_times, *compartment.shape)
    """
    ys = jnp.asarray(solution.ys)

    if ys.shape[-1] != runtime.total_state_size:
        raise OdeSolverError(
            "solution.ys does not have runtime.total_state_size as its final "
            f"dimension. Expected {runtime.total_state_size}, got {ys.shape[-1]}."
        )

    leading_shape = tuple(ys.shape[:-1])
    result: dict[str, Any] = {}

    for compartment in runtime.state_layout.compartments:
        flat_piece = ys[..., compartment.start : compartment.stop]
        result[compartment.name] = jnp.reshape(
            flat_piece,
            leading_shape + compartment.shape,
        )

    return result


def solution_final_state_flat(
    *,
    solution: dfx.Solution,
) -> Any:
    """
    Return the final saved flat state from a Diffrax solution.
    """
    ys = jnp.asarray(solution.ys)

    if ys.ndim == 1:
        return ys

    return ys[-1]


def solution_final_state_dict(
    *,
    runtime: RuntimeModel,
    solution: dfx.Solution,
) -> dict[str, Any]:
    """
    Return the final saved state as a compartment dictionary.
    """
    final_flat = solution_final_state_flat(solution=solution)

    return runtime.state_layout.unflatten(final_flat)


__all__ = [
    "ParameterContext",
    "FlatState",
    "StateDict",
    "RHSFn",
    "RHSStateFormat",
    "RHSCallStyle",
    "OdeSolverError",
    "OdeSolverOptions",
    "solve_ode",
    "build_diffeqsolve_kwargs",
    "make_ode_term",
    "make_vector_field",
    "prepare_initial_state_for_solve",
    "resolve_time_span",
    "normalize_rhs_output",
    "validate_solution",
    "solution_ys_as_state_dict",
    "solution_final_state_flat",
    "solution_final_state_dict",
]
