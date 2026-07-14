from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import diffrax as dfx
import jax.numpy as jnp

from dynode.runtime.layout.runtime_model import RuntimeModel

from .errors import OdeSolverError
from .ode_options import OdeSolverOptions


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
