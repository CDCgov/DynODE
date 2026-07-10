from __future__ import annotations

from typing import Any


def dependency_set(obj: Any, attr_name: str) -> set[str]:
    attr = getattr(obj, attr_name, None)

    if attr is None:
        return set()

    if callable(attr):
        return set(attr())

    return set(attr)


def strain_parameter_values(strain: Any) -> tuple[Any, ...]:
    values: list[Any] = [
        strain.r0,
        strain.infectious_period,
    ]

    optional_values = (
        strain.exposed_to_infectious,
        strain.introduction_time,
        strain.introduction_percentage,
        strain.introduction_scale,
    )

    for value in optional_values:
        if value is not None:
            values.append(value)

    return tuple(values)


def strain_parameter_dependencies(strain: Any) -> set[str]:
    deps: set[str] = set()

    for value in strain_parameter_values(strain):
        deps |= value.parameter_dependencies()

    for interaction in strain.interactions.values():
        deps |= dependency_set(interaction, "parameter_dependencies")

    return deps


def strain_deterministic_dependencies(strain: Any) -> set[str]:
    deps: set[str] = set()

    for value in strain_parameter_values(strain):
        deps |= value.deterministic_dependencies()

    for interaction in strain.interactions.values():
        deps |= dependency_set(interaction, "deterministic_dependencies")

    return deps


def strain_data_dependencies(strain: Any) -> set[str]:
    deps: set[str] = set()

    for value in strain_parameter_values(strain):
        deps |= value.data_dependencies()

    for interaction in strain.interactions.values():
        deps |= dependency_set(interaction, "data_dependencies")

    return deps
