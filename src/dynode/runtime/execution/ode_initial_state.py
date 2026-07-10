from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp

from dynode.runtime.layout.runtime_model import RuntimeModel

from .errors import OdeSolverError
from .ode_options import OdeSolverOptions
from .types import FlatState, StateMapping


def prepare_initial_state_for_solve(
    *,
    runtime: RuntimeModel,
    y0: FlatState | StateMapping,
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