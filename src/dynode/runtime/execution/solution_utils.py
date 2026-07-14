from __future__ import annotations

from typing import Any

import diffrax as dfx
import jax.numpy as jnp

from dynode.runtime.layout.runtime_model import RuntimeModel

from .errors import OdeSolverError


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
