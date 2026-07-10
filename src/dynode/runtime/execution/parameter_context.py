from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dynode.runtime.layout.runtime_model import RuntimeModel

from .deterministic_resolution import resolve_deterministic_parameters
from .errors import ParameterSamplingError
from .parameter_options import ParameterSamplingOptions
from .types import ParameterContext


def make_initial_context(
    *,
    initial_context: Mapping[str, Any] | None,
) -> ParameterContext:
    if initial_context is None:
        return {}

    return dict(initial_context)


def evaluate_parameter_context_without_numpyro(
    *,
    runtime: RuntimeModel,
    sampled_values: Mapping[str, Any],
    data: Any | None = None,
    options: ParameterSamplingOptions | None = None,
) -> ParameterContext:
    """
    Build a full parameter context from externally supplied sampled values.

    This is useful for:
    - unit tests
    - deterministic simulations
    - posterior predictive simulation from existing samples
    - debugging deterministic expressions outside NumPyro

    This function does not call numpyro.sample.
    """
    options = options or ParameterSamplingOptions(
        record_deterministics=False,
    )

    context: ParameterContext = dict(sampled_values)

    missing_priors = sorted(
        set(runtime.parameter_layout.prior_names) - set(context)
    )

    if missing_priors:
        raise ParameterSamplingError(
            "sampled_values is missing required prior values: "
            f"{missing_priors}."
        )

    resolve_deterministic_parameters(
        parameter_layout=runtime.parameter_layout,
        context=context,
        data=data,
        options=options,
    )

    runtime.validate_parameter_context(
        context,
        require_all=True,
    )

    return context


def split_parameter_context(
    *,
    runtime: RuntimeModel,
    context: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """
    Split a full parameter context into sampled and deterministic components.
    """
    sampled = {
        name: context[name]
        for name in runtime.parameter_layout.prior_names
        if name in context
    }

    deterministic = {
        name: context[name]
        for name in runtime.parameter_layout.deterministic_names
        if name in context
    }

    return {
        "sampled": sampled,
        "deterministic": deterministic,
    }