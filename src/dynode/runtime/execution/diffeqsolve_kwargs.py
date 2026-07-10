from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dynode.runtime.layout.runtime_model import RuntimeModel

from .errors import OdeSolverError


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
        validate_extra_diffeqsolve_kwargs(extra_kwargs)
        kwargs.update(dict(extra_kwargs))

    return kwargs


def validate_extra_diffeqsolve_kwargs(
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