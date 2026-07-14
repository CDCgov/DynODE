from __future__ import annotations

from collections.abc import Mapping
from typing import Any


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


def available_data_names(data: Any) -> set[str]:
    for attr_name in (
        "observation_names",
        "observed_series_names",
        "data_names",
    ):
        value = getattr(data, attr_name, None)

        if value is None:
            continue

        if callable(value):
            value = value()

        return {str(name) for name in value}

    if isinstance(data, Mapping):
        observations = data.get("observations")

        if isinstance(observations, Mapping):
            return {str(name) for name in observations}

        return {str(name) for name in data}

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
