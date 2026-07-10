from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from typing import Any

from dynode.runtime.layout.runtime_model import RuntimeModel

from .errors import OdeSolverError
from .types import ParameterMapping, RHSCallStyle, RHSFn


def make_rhs_adapter(
    *,
    rhs_fn: RHSFn,
    call_style: RHSCallStyle,
) -> Callable[..., Any]:
    if call_style == "standard":
        return make_standard_rhs_adapter(rhs_fn)

    if call_style == "keyword":
        return make_keyword_rhs_adapter(rhs_fn)

    if call_style != "auto":
        raise OdeSolverError(
            f"Unknown rhs_call_style {call_style!r}. "
            "Expected 'auto', 'standard', or 'keyword'."
        )

    signature = inspect.signature(rhs_fn)
    parameter_names = set(signature.parameters)

    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )

    keyword_names = {
        "params",
        "runtime",
        "data",
        "state_layout",
        "transmission",
        "extra",
    }

    if accepts_kwargs or parameter_names & keyword_names:
        return make_keyword_rhs_adapter(rhs_fn)

    return make_standard_rhs_adapter(rhs_fn)


def make_standard_rhs_adapter(
    rhs_fn: RHSFn,
) -> Callable[..., Any]:
    """
    Adapter for RHS functions of shape:

        rhs_fn(t, y, args)

    where args is a dictionary containing params, runtime, data, and helpers.
    """

    def adapter(
        *,
        t: Any,
        y: Any,
        params: ParameterMapping,
        runtime: RuntimeModel,
        data: Any | None,
        extra: Mapping[str, Any],
    ) -> Any:
        rhs_args = {
            "params": params,
            "runtime": runtime,
            "data": data,
            "state_layout": runtime.state_layout,
            "transmission": runtime.transmission,
            "extra": extra,
        }

        return rhs_fn(t, y, rhs_args)

    return adapter


def make_keyword_rhs_adapter(
    rhs_fn: RHSFn,
) -> Callable[..., Any]:
    """
    Adapter for RHS functions using named arguments.

    Supported examples:
        rhs_fn(t=t, y=y, params=params, runtime=runtime)
        rhs_fn(t, y, params=params, runtime=runtime)
        rhs_fn(t, y, *, params, runtime, data=None)

    Important: do not catch broad exceptions around the RHS call. If the RHS
    raises a real tracing/runtime error, catching it and retrying with a
    different signature obscures the true root cause. Select the call style from
    the function signature before invoking the RHS.
    """
    signature = inspect.signature(rhs_fn)

    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )

    def can_pass_as_keyword(parameter_name: str) -> bool:
        parameter = signature.parameters.get(parameter_name)
        if parameter is None:
            return False
        return parameter.kind in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }

    positional_parameters = tuple(
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind
        in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        }
    )

    pass_t_y_by_keyword = can_pass_as_keyword("t") and can_pass_as_keyword(
        "y"
    )
    pass_t_y_positionally = (
        not pass_t_y_by_keyword and len(positional_parameters) >= 2
    )

    def adapter(
        *,
        t: Any,
        y: Any,
        params: ParameterMapping,
        runtime: RuntimeModel,
        data: Any | None,
        extra: Mapping[str, Any],
    ) -> Any:
        kwargs = {
            "t": t,
            "y": y,
            "params": params,
            "runtime": runtime,
            "data": data,
            "state_layout": runtime.state_layout,
            "transmission": runtime.transmission,
            "extra": extra,
        }

        if accepts_kwargs:
            supported_kwargs = kwargs
        else:
            supported_kwargs = {
                key: value
                for key, value in kwargs.items()
                if key in signature.parameters
            }

        if pass_t_y_by_keyword:
            return rhs_fn(**supported_kwargs)

        if pass_t_y_positionally:
            supported_kwargs = dict(supported_kwargs)
            supported_kwargs.pop("t", None)
            supported_kwargs.pop("y", None)
            return rhs_fn(t, y, **supported_kwargs)

        return rhs_fn(**supported_kwargs)

    return adapter