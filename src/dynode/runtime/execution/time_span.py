from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from dynode.runtime.layout.runtime_model import RuntimeModel

from .errors import OdeSolverError


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
        extract_time_values(data)
        or extract_time_values(runtime.data_spec)
        or extract_time_values_from_solver_spec(runtime.solver_spec)
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


def extract_time_values(source: Any | None) -> tuple[float, ...] | None:
    if source is None:
        return None

    if isinstance(source, Mapping):
        if "time" not in source:
            return None

        return coerce_time_values(source["time"])

    time_obj = getattr(source, "time", None)

    if time_obj is None:
        return None

    return coerce_time_values(time_obj)


def coerce_time_values(time_obj: Any) -> tuple[float, ...] | None:
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


def extract_time_values_from_solver_spec(
    solver_spec: Any,
) -> tuple[float, ...] | None:
    save_at = getattr(solver_spec, "save_at", None)

    if save_at is None:
        return None

    ts = getattr(save_at, "ts", None)

    if ts is None:
        return None

    return tuple(float(value) for value in ts)