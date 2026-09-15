from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

import jax.numpy as jnp
import numpyro

from dynode.runtime.layout.runtime_model import RuntimeParameterLayout

from .errors import ParameterSamplingError
from .parameter_options import ParameterSamplingOptions
from .parameter_utils import (
    deterministic_site_name,
    evaluate_deterministic,
    name_of,
    validate_dependencies_available,
)


def record_parameter_debug(
    *,
    name: str,
    value: Any,
    prefix: str | None = None,
) -> None:
    """Record finite-value diagnostics for sampled or deterministic parameters."""
    site_name = f"{prefix}_{name}" if prefix else name

    if value is None:
        return

    try:
        arr = jnp.asarray(value)
    except Exception:
        return

    finite = jnp.isfinite(arr)

    safe_arr = jnp.nan_to_num(
        arr,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    numpyro.deterministic(
        f"debug_param_{site_name}_has_nan",
        jnp.any(jnp.isnan(arr)),
    )
    numpyro.deterministic(
        f"debug_param_{site_name}_has_inf",
        jnp.any(jnp.isinf(arr)),
    )
    numpyro.deterministic(
        f"debug_param_{site_name}_finite_frac",
        jnp.mean(finite.astype(jnp.float32)),
    )
    numpyro.deterministic(
        f"debug_param_{site_name}_min",
        jnp.min(safe_arr),
    )
    numpyro.deterministic(
        f"debug_param_{site_name}_max",
        jnp.max(safe_arr),
    )


def resolve_deterministic_parameters(
    *,
    parameter_layout: RuntimeParameterLayout,
    context: MutableMapping[str, Any],
    data: Any | None = None,
    options: ParameterSamplingOptions | None = None,
) -> MutableMapping[str, Any]:
    """
    Resolve deterministic parameters in topological order.

    The deterministic order is compiled by ParameterSpec / compile_model.py.
    """
    options = options or ParameterSamplingOptions()

    for deterministic in parameter_layout.deterministic_order:
        resolve_deterministic_parameter(
            deterministic=deterministic,
            context=context,
            data=data,
            options=options,
        )

    return context


def resolve_deterministic_parameter(
    *,
    deterministic: Any,
    context: MutableMapping[str, Any],
    data: Any | None = None,
    options: ParameterSamplingOptions | None = None,
) -> Any:
    """
    Evaluate one DeterministicSpec and write the value into context.
    """
    options = options or ParameterSamplingOptions()

    deterministic_name = name_of(deterministic)

    if (
        deterministic_name in context
        and not options.allow_initial_context_overwrite
    ):
        raise ParameterSamplingError(
            f"Parameter context already contains deterministic parameter "
            f"{deterministic_name!r}. Set allow_initial_context_overwrite=True "
            "only if this is intentional."
        )

    if options.validate_dependencies:
        validate_dependencies_available(
            obj=deterministic,
            context=context,
            obj_label=f"Deterministic parameter {deterministic_name!r}",
        )

    value = evaluate_deterministic(
        deterministic=deterministic,
        context=context,
        data=data,
    )

    if options.record_deterministics:
        value = numpyro.deterministic(
            deterministic_site_name(deterministic, options.scope),
            value,
        )
        record_parameter_debug(
            name=deterministic_site_name(deterministic, options.scope),
            value=value,
            prefix=None,
        )

    context[deterministic_name] = value

    return value
