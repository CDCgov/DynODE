from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any

from .errors import ParameterSamplingError


def name_of(obj: Any) -> str:
    name = getattr(obj, "name", None)

    if name is None:
        raise ParameterSamplingError(
            f"Expected object {obj!r} to expose a 'name' attribute."
        )

    return str(name)


def dependency_set(
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


def call_with_supported_kwargs(
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


def validate_dependencies_available(
    *,
    obj: Any,
    context: Mapping[str, Any],
    obj_label: str,
) -> None:
    dependencies = dependency_set(obj, "dependencies")
    missing = sorted(dependencies - set(context))

    if missing:
        raise ParameterSamplingError(
            f"{obj_label} depends on values that are not available yet: "
            f"{missing}. Available context values are: {sorted(context)}."
        )


def prior_to_numpyro(
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
                f"Prior {name_of(prior)!r} does not expose to_numpyro(...) "
                "or a distribution attribute."
            )

        to_numpyro = getattr(distribution, "to_numpyro", None)

        if not callable(to_numpyro):
            raise ParameterSamplingError(
                f"Prior {name_of(prior)!r} distribution does not expose "
                "to_numpyro(...)."
            )

    try:
        return call_with_supported_kwargs(
            to_numpyro,
            context=context,
            data=data,
        )
    except Exception as exc:
        raise ParameterSamplingError(
            f"Failed to construct NumPyro distribution for prior "
            f"{name_of(prior)!r}."
        ) from exc


def evaluate_deterministic(
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
            return call_with_supported_kwargs(
                evaluate,
                context=context,
                data=data,
            )
        except Exception as exc:
            raise ParameterSamplingError(
                f"Failed to evaluate deterministic parameter "
                f"{name_of(deterministic)!r}."
            ) from exc

    expression = getattr(deterministic, "expression", None)

    if expression is None:
        raise ParameterSamplingError(
            f"Deterministic parameter {name_of(deterministic)!r} does not "
            "expose evaluate(...) or expression.evaluate(...)."
        )

    expression_evaluate = getattr(expression, "evaluate", None)

    if not callable(expression_evaluate):
        raise ParameterSamplingError(
            f"Expression for deterministic parameter {name_of(deterministic)!r} "
            "does not expose evaluate(...)."
        )

    try:
        return call_with_supported_kwargs(
            expression_evaluate,
            context=context,
            data=data,
        )
    except Exception as exc:
        raise ParameterSamplingError(
            f"Failed to evaluate expression for deterministic parameter "
            f"{name_of(deterministic)!r}."
        ) from exc


def sample_site_name(prior: Any, scope: str | None) -> str:
    sample_site_name_fn = getattr(prior, "sample_site_name", None)
    if callable(sample_site_name_fn):
        return str(sample_site_name_fn(scope=scope))
    name = name_of(prior)
    return f"{scope}_{name}" if scope else name


def deterministic_site_name(deterministic: Any, scope: str | None) -> str:
    name = name_of(deterministic)
    return f"{scope}_{name}" if scope else name
