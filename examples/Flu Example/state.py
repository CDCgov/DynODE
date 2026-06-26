from __future__ import annotations

from typing import Any, Mapping

import jax.numpy as jnp

from dynode.runtime.runtime_model import RuntimeModel

from .rhs import wane_immunity


def apply_flu_escape_initial_state(
    *,
    y0: Any,
    context: Any,
    runtime: RuntimeModel,
    params: Mapping[str, Any],
    data: Any | None = None,
) -> Any:
    """Parameter-dependent initial-state transform for immune escape."""
    state = (
        runtime.state_layout.unflatten(y0)
        if not isinstance(y0, Mapping)
        else dict(y0)
    )
    s = state["s"]
    ds = jnp.zeros_like(s)
    ds = wane_immunity(
        s,
        ds,
        params.get("escape_h1", 0.0),
        params.get("escape_h3", 0.0),
        params.get("escape_b", 0.0),
    )
    state["s"] = s + ds
    return runtime.state_layout.flatten(state, allow_broadcast=False)
