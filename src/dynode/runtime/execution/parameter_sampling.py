from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dynode.runtime.layout.runtime_model import (
    RuntimeModel,
    RuntimeParameterLayout,
)

from .deterministic_resolution import resolve_deterministic_parameters
from .parameter_context import make_initial_context
from .parameter_options import ParameterSamplingOptions
from .prior_sampling import sample_prior_parameters
from .types import ParameterContext


def sample_parameters(
    *,
    runtime: RuntimeModel,
    data: Any | None = None,
    initial_context: Mapping[str, Any] | None = None,
    options: ParameterSamplingOptions | None = None,
) -> ParameterContext:
    """
    Sample all prior parameters and resolve all deterministic parameters.

    This is the main entry point used by DynodeModel.

    Parameters
    ----------
    runtime:
        Compiled RuntimeModel.

    data:
        Optional observed/model data object. Deterministic expressions may use
        this if they contain DataRef values.

    initial_context:
        Optional values to seed the context with before sampling. This is useful
        for advanced cases, but most models should leave it as None.

    options:
        Sampling behavior options.

    Returns
    -------
    dict[str, Any]
        Full parameter context containing sampled and deterministic parameters.

    Notes
    -----
    This function should be called inside a NumPyro model trace because it calls
    numpyro.sample and numpyro.deterministic.
    """
    options = options or ParameterSamplingOptions()

    context = make_initial_context(
        initial_context=initial_context,
    )

    sample_prior_parameters(
        parameter_layout=runtime.parameter_layout,
        context=context,
        data=data,
        options=options,
    )

    resolve_deterministic_parameters(
        parameter_layout=runtime.parameter_layout,
        context=context,
        data=data,
        options=options,
    )

    if options.validate_dependencies:
        runtime.validate_parameter_context(
            context,
            require_all=True,
        )

    return context


def sample_parameter_layout(
    *,
    parameter_layout: RuntimeParameterLayout,
    data: Any | None = None,
    initial_context: Mapping[str, Any] | None = None,
    scope: str | None = None,
    options: ParameterSamplingOptions | None = None,
) -> ParameterContext:
    """Sample a compiled RuntimeParameterLayout outside a full RuntimeModel.

    This is used by ExperimentRuntime for shared and instance-local parameter
    blocks. The returned context is keyed by logical parameter names, while
    NumPyro sample sites are scoped.
    """
    base_options = options or ParameterSamplingOptions()
    scoped_options = ParameterSamplingOptions(
        validate_dependencies=base_options.validate_dependencies,
        record_deterministics=base_options.record_deterministics,
        allow_initial_context_overwrite=base_options.allow_initial_context_overwrite,
        scope=scope if scope is not None else base_options.scope,
        metadata=base_options.metadata,
    )
    context = make_initial_context(initial_context=initial_context)
    sample_prior_parameters(
        parameter_layout=parameter_layout,
        context=context,
        data=data,
        options=scoped_options,
    )
    resolve_deterministic_parameters(
        parameter_layout=parameter_layout,
        context=context,
        data=data,
        options=scoped_options,
    )
    return context