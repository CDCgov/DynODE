from __future__ import annotations

from typing import Any

from dynode.runtime.layout.runtime_model import (
    RuntimeParameterLayout,
    RuntimeTransmission,
    StateLayout,
)

from .errors import CompileError
from .options import CompileOptions
from .ordering import deterministic_execution_order, prior_execution_order
from .utils import get_sequence_attr, name_of


def compile_state_layout(simulation: Any) -> StateLayout:
    """
    Compile SimulationSpec into StateLayout.
    """
    try:
        return StateLayout.from_simulation(simulation)
    except Exception as exc:
        raise CompileError(
            "Failed to compile StateLayout from SimulationSpec."
        ) from exc


def compile_parameter_layout(
    parameters: Any,
    *,
    options: CompileOptions,
) -> RuntimeParameterLayout:
    """
    Compile ParameterBlockSpec-like input into RuntimeParameterLayout.

    Priors are topologically ordered when prior distributions depend on
    previously sampled priors or on externally supplied/shared parameters.
    """
    priors = get_sequence_attr(
        parameters,
        ("priors", "prior_specs"),
    )

    deterministic = get_sequence_attr(
        parameters,
        ("deterministic", "deterministics", "deterministic_params"),
    )

    prior_order = prior_execution_order(
        priors,
        deterministic=deterministic,
        options=options,
    )

    deterministic_order = deterministic_execution_order(
        parameters,
        deterministic,
    )

    return RuntimeParameterLayout(
        prior_names=tuple(name_of(prior) for prior in prior_order),
        deterministic_names=tuple(name_of(spec) for spec in deterministic),
        deterministic_order=tuple(deterministic_order),
        prior_specs={name_of(prior): prior for prior in prior_order},
        deterministic_specs={name_of(spec): spec for spec in deterministic},
    )


def compile_transmission(
    transmission: Any,
    *,
    simulation: Any,
) -> RuntimeTransmission:
    """
    Compile TransmissionSpec into RuntimeTransmission.
    """
    age_bins = age_bins_from_simulation(simulation)

    try:
        return RuntimeTransmission.from_spec(
            transmission,
            age_bins=age_bins,
        )
    except Exception as exc:
        raise CompileError(
            "Failed to compile RuntimeTransmission from TransmissionSpec."
        ) from exc


def age_bins_from_simulation(simulation: Any) -> tuple[Any, ...]:
    get_age_bins = getattr(simulation, "get_age_bins", None)

    if callable(get_age_bins):
        return tuple(get_age_bins())

    age_bins: list[Any] = []

    flatten_unique_dims = getattr(simulation, "flatten_unique_dims", None)

    if callable(flatten_unique_dims):
        for dimension in flatten_unique_dims():
            bins = tuple(getattr(dimension, "bins", ()))

            if bins and all(
                bin_.__class__.__name__ == "AgeBin" for bin_ in bins
            ):
                age_bins.extend(bins)
                break

    return tuple(age_bins)