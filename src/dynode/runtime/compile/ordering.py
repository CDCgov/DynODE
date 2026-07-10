from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .errors import CompileError
from .options import CompileOptions
from .utils import dependency_set, duplicates, name_of


def prior_execution_order(
    priors: Sequence[Any],
    *,
    deterministic: Sequence[Any],
    options: CompileOptions,
) -> tuple[Any, ...]:
    """
    Determine prior sampling order.

    The default case is simple: priors have no dependencies, so the original
    order is preserved.

    If a prior distribution depends on another sampled prior, this function
    orders priors topologically. Dependencies on names supplied by
    CompileOptions.external_parameter_names are treated as already resolved.
    """
    if not priors:
        return tuple()

    prior_names = tuple(name_of(prior) for prior in priors)
    deterministic_names = tuple(name_of(spec) for spec in deterministic)

    duplicate_names = duplicates(prior_names)

    if duplicate_names:
        raise CompileError(f"Duplicate prior names: {duplicate_names}.")

    prior_name_set = set(prior_names)
    deterministic_name_set = set(deterministic_names)
    external_name_set = {
        str(name) for name in options.external_parameter_names
    }

    prior_by_name = {name_of(prior): prior for prior in priors}

    for prior_name, prior in prior_by_name.items():
        deps = dependency_set(prior, "dependencies")

        if prior_name in deps:
            raise CompileError(
                f"Prior {prior_name!r} cannot depend on itself."
            )

        deterministic_deps = deps & deterministic_name_set

        if (
            deterministic_deps
            and not options.allow_prior_dependencies_on_deterministics
        ):
            raise CompileError(
                f"Prior {prior_name!r} depends on deterministic parameters "
                f"{sorted(deterministic_deps)}, but deterministic parameters "
                "are resolved after priors in the current runtime design."
            )

        unknown = (
            deps - prior_name_set - deterministic_name_set - external_name_set
        )

        if unknown:
            raise CompileError(
                f"Prior {prior_name!r} depends on unknown parameters "
                f"{sorted(unknown)}."
            )

    remaining = dict(prior_by_name)

    resolved: set[str] = set(external_name_set)

    if options.allow_prior_dependencies_on_deterministics:
        resolved |= deterministic_name_set

    ordered: list[Any] = []

    while remaining:
        ready_names = [
            name
            for name in prior_names
            if name in remaining
            and dependency_set(remaining[name], "dependencies") <= resolved
        ]

        if not ready_names:
            unresolved = {
                name: sorted(dependency_set(prior, "dependencies") - resolved)
                for name, prior in remaining.items()
            }

            raise CompileError(
                "Could not determine prior sampling order. This usually means "
                "there is a cyclic prior dependency or a prior depends on a "
                "deterministic value that is resolved after priors. "
                f"Remaining dependencies: {unresolved}."
            )

        for name in ready_names:
            prior = remaining.pop(name)
            ordered.append(prior)
            resolved.add(name)

    return tuple(ordered)


def deterministic_execution_order(
    parameters: Any,
    deterministic: Sequence[Any],
) -> tuple[Any, ...]:
    deterministic_execution_order_fn = getattr(
        parameters,
        "deterministic_execution_order",
        None,
    )

    if callable(deterministic_execution_order_fn):
        try:
            return tuple(deterministic_execution_order_fn())
        except Exception as exc:
            raise CompileError(
                "Failed to determine deterministic parameter execution order."
            ) from exc

    return tuple(deterministic)
