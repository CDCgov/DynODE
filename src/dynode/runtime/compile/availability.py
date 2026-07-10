from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dynode.runtime.layout.runtime_model import RuntimeModel

from .options import CompileOptions


def available_parameter_names(
    runtime: RuntimeModel,
    *,
    options: CompileOptions | None = None,
) -> set[str]:
    """
    Names available for resolving parameter references during compilation.

    Includes:
    - model-local prior names
    - model-local deterministic parameter names
    - optional external/shared names supplied by an outer experiment context
    """
    parameter_layout = runtime.parameter_layout
    names: set[str] = set()

    for attr_name in (
        "prior_names",
        "deterministic_names",
        "resolved_names",
        "resolved_parameter_names",
    ):
        value = getattr(parameter_layout, attr_name, None)

        if value is None:
            continue

        if callable(value):
            value = value()

        names |= {str(name) for name in value}

    if options is not None:
        names |= {str(name) for name in options.external_parameter_names}

    return names


def available_data_names(data: Any | None) -> set[str]:
    if data is None:
        return set()

    if isinstance(data, Mapping):
        names = set(data)

        observations = data.get("observations")
        if isinstance(observations, Mapping):
            names |= set(observations)

        covariates = data.get("covariates")
        if isinstance(covariates, Mapping):
            names |= set(covariates)

        return {str(name) for name in names}

    for attr_name in (
        "field_names",
        "observation_names",
        "observed_series_names",
        "data_names",
        "covariate_names",
        "index_names",
    ):
        value = getattr(data, attr_name, None)

        if value is None:
            continue

        if callable(value):
            value = value()

        return {str(name) for name in value}

    fields = getattr(data, "fields", None)

    if fields is not None:
        names = {
            str(getattr(field, "name"))
            for field in fields
            if getattr(field, "name", None) is not None
        }
        if names:
            return names

    observations = getattr(data, "observations", None)

    if observations is None:
        return set()

    if isinstance(observations, Mapping):
        return {str(name) for name in observations}

    names: set[str] = set()

    for observation in observations:
        name = getattr(observation, "name", None)

        if name is not None:
            names.add(str(name))

    return names