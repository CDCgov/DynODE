from __future__ import annotations

import inspect
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass, field
from typing import Any

import numpyro

from .runtime_model import RuntimeModel, RuntimeParameterLayout


ParameterContext = dict[str, Any]

class ParameterSamplingError(RuntimeError):
    """
    Raised when parameters cannot be sampled or deterministic parameters cannot
    be resolved.
    """

@dataclass(frozen=True, slots=True)
class ParameterSamplingOptions:
    """
    Options controlling parameter sampling and deterministic resolution.
    """

    validate_dependencies: bool = True

    record_deterministics: bool = True

    allow_initial_context_overwrite: bool = False

    metadata: Mapping[str, str] = field(default_factory=dict)

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

    context = _make_initial_context(
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

def sample_prior_parameters(
    *,
    parameter_layout: RuntimeParameterLayout,
    context: MutableMapping[str, Any],
    data: Any | None = None,
    options: ParameterSamplingOptions | None = None,
) -> MutableMapping[str, Any]:
    """
    Sample all prior parameters in the compiled prior order.

    Prior order is determined by compile_model.py. This allows priors to depend
    on previously sampled priors when needed.
    """
    options = options or ParameterSamplingOptions()

    for prior_name in parameter_layout.prior_names:
        prior = parameter_layout.get_prior(prior_name)

        sample_prior_parameter(
            prior=prior,
            context=context,
            data=data,
            options=options,
        )

    return context

def sample_prior_parameter(
    *,
    prior: Any,
    context: MutableMapping[str, Any],
    data: Any | None = None,
    options: ParameterSamplingOptions | None = None,
) -> Any:
    """
    Sample a single PriorSpec and write the sampled value into context.
    """
    options = options or ParameterSamplingOptions()

    prior_name = _name_of(prior)

    if prior_name in context and not options.allow_initial_context_overwrite:
        raise ParameterSamplingError(
            f"Parameter context already contains prior {prior_name!r}. "
            "Set allow_initial_context_overwrite=True only if this is intentional."
        )

    if options.validate_dependencies:
        _validate_dependencies_available(
            obj=prior,
            context=context,
            obj_label=f"Prior {prior_name!r}",
        )

    distribution = _prior_to_numpyro(
        prior=prior,
        context=context,
        data=data,
    )

    value = numpyro.sample(
        prior_name,
        distribution,
    )

    context[prior_name] = value

    return value

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

    deterministic_name = _name_of(deterministic)

    if deterministic_name in context and not options.allow_initial_context_overwrite:
        raise ParameterSamplingError(
            f"Parameter context already contains deterministic parameter "
            f"{deterministic_name!r}. Set allow_initial_context_overwrite=True "
            "only if this is intentional."
        )

    if options.validate_dependencies:
        _validate_dependencies_available(
            obj=deterministic,
            context=context,
            obj_label=f"Deterministic parameter {deterministic_name!r}",
        )

    value = _evaluate_deterministic(
        deterministic=deterministic,
        context=context,
        data=data,
    )

    if options.record_deterministics:
        value = numpyro.deterministic(
            deterministic_name,
            value,
        )

    context[deterministic_name] = value

    return value

def _make_initial_context(
    *,
    initial_context: Mapping[str, Any] | None,
) -> ParameterContext:
    if initial_context is None:
        return {}

    return dict(initial_context)

def _prior_to_numpyro(
    *,
    prior: Any,
    context: Mapping[str, Any],
    data: Any | None = None,
) -> Any:
    """
    Convert a PriorSpec into a NumPyro distribution.

    Preferred API:

        prior.to_numpyro(context=context)

    This helper also supports a future extension:

        prior.to_numpyro(context=context, data=data)
    """
    to_numpyro = getattr(prior, "to_numpyro", None)

    if not callable(to_numpyro):
        distribution = getattr(prior, "distribution", None)

        if distribution is None:
            raise ParameterSamplingError(
                f"Prior {_name_of(prior)!r} does not expose to_numpyro(...) "
                "or a distribution attribute."
            )

        to_numpyro = getattr(distribution, "to_numpyro", None)

        if not callable(to_numpyro):
            raise ParameterSamplingError(
                f"Prior {_name_of(prior)!r} distribution does not expose "
                "to_numpyro(...)."
            )

    try:
        return _call_with_supported_kwargs(
            to_numpyro,
            context=context,
            data=data,
        )
    except Exception as exc:
        raise ParameterSamplingError(
            f"Failed to construct NumPyro distribution for prior "
            f"{_name_of(prior)!r}."
        ) from exc

def _evaluate_deterministic(
    *,
    deterministic: Any,
    context: Mapping[str, Any],
    data: Any | None = None,
) -> Any:
    """
    Evaluate a DeterministicSpec.

    Preferred API:

        deterministic.evaluate(context=context, data=data)

    Backward-compatible API:

        deterministic.evaluate(context=context)

    Fallback:

        deterministic.expression.evaluate(context=context, data=data)
    """
    evaluate = getattr(deterministic, "evaluate", None)

    if callable(evaluate):
        try:
            return _call_with_supported_kwargs(
                evaluate,
                context=context,
                data=data,
            )
        except Exception as exc:
            raise ParameterSamplingError(
                f"Failed to evaluate deterministic parameter "
                f"{_name_of(deterministic)!r}."
            ) from exc

    expression = getattr(deterministic, "expression", None)

    if expression is None:
        raise ParameterSamplingError(
            f"Deterministic parameter {_name_of(deterministic)!r} does not "
            "expose evaluate(...) or expression.evaluate(...)."
        )

    expression_evaluate = getattr(expression, "evaluate", None)

    if not callable(expression_evaluate):
        raise ParameterSamplingError(
            f"Expression for deterministic parameter {_name_of(deterministic)!r} "
            "does not expose evaluate(...)."
        )

    try:
        return _call_with_supported_kwargs(
            expression_evaluate,
            context=context,
            data=data,
        )
    except Exception as exc:
        raise ParameterSamplingError(
            f"Failed to evaluate expression for deterministic parameter "
            f"{_name_of(deterministic)!r}."
        ) from exc

def _validate_dependencies_available(
    *,
    obj: Any,
    context: Mapping[str, Any],
    obj_label: str,
) -> None:
    dependencies = _dependency_set(obj, "dependencies")
    missing = sorted(dependencies - set(context))

    if missing:
        raise ParameterSamplingError(
            f"{obj_label} depends on values that are not available yet: "
            f"{missing}. Available context values are: {sorted(context)}."
        )

def _dependency_set(
    obj: Any,
    attr_name: str,
) -> set[str]:
    attr = getattr(obj, attr_name, None)

    if attr is None:
        return set()

    if callable(attr):
        value = attr()
    else:
        value = attr

    if value is None:
        return set()

    return {str(item) for item in value}

def _name_of(obj: Any) -> str:
    name = getattr(obj, "name", None)

    if name is None:
        raise ParameterSamplingError(
            f"Expected object {obj!r} to expose a 'name' attribute."
        )

    return str(name)

def _call_with_supported_kwargs(
    fn: Any,
    **kwargs: Any,
) -> Any:
    """
    Call a function/method with only the keyword arguments it supports.

    This avoids fragile try/except TypeError logic while allowing specs to
    evolve from:

        evaluate(context=...)

    to:

        evaluate(context=..., data=...)
    """
    signature = inspect.signature(fn)

    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )

    if accepts_kwargs:
        return fn(**kwargs)

    supported_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters
    }

    return fn(**supported_kwargs)

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

__all__ = [
    "ParameterContext",
    "ParameterSamplingError",
    "ParameterSamplingOptions",
    "sample_parameters",
    "sample_prior_parameters",
    "sample_prior_parameter",
    "resolve_deterministic_parameters",
    "resolve_deterministic_parameter",
    "evaluate_parameter_context_without_numpyro",
    "split_parameter_context",
]