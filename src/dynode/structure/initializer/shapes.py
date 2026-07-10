from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np


def coerce_to_shape(
    value: Any,
    target_shape: tuple[int, ...],
    compartment_name: str,
    allow_broadcast: bool,
) -> Any:
    array = jnp.asarray(value)
    value_shape = tuple(array.shape)

    if value_shape == target_shape:
        return array

    if allow_broadcast:
        try:
            return jnp.broadcast_to(array, target_shape)
        except ValueError as exc:
            raise ValueError(
                f"Initial value for compartment {compartment_name!r} has shape "
                f"{value_shape}, which cannot be broadcast to {target_shape}."
            ) from exc

    raise ValueError(
        f"Initial value for compartment {compartment_name!r} has shape "
        f"{value_shape}, but expected {target_shape}."
    )


def shape_matches(
    value_shape: tuple[int, ...],
    target_shape: tuple[int, ...],
) -> bool:
    return value_shape == target_shape


def can_broadcast(
    value_shape: tuple[int, ...],
    target_shape: tuple[int, ...],
) -> bool:
    try:
        broadcast_shape = np.broadcast_shapes(value_shape, target_shape)
    except ValueError:
        return False

    return tuple(broadcast_shape) == target_shape
