from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import diffrax as dfx

from dynode.runtime.layout.runtime_model import RuntimeModel

from .diffeqsolve_kwargs import build_diffeqsolve_kwargs
from .errors import OdeSolverError
from .ode_initial_state import prepare_initial_state_for_solve
from .ode_options import OdeSolverOptions
from .ode_terms import make_ode_term
from .ode_validation import validate_solution
from .time_span import resolve_time_span
from .types import FlatState, ParameterMapping, RHSFn, StateMapping


def solve_ode(
    *,
    runtime: RuntimeModel,
    rhs_fn: RHSFn,
    y0: FlatState | StateMapping,
    params: ParameterMapping,
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
