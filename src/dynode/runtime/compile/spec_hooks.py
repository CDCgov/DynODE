from __future__ import annotations

from typing import Any

from .errors import CompileError
from .options import CompileOptions


def validate_required_model_shape(spec: Any) -> None:
    required_top_level = ("simulation", "parameters", "solver", "transmission")

    for attr_name in required_top_level:
        if not hasattr(spec, attr_name):
            raise CompileError(
                f"Model spec is missing required attribute {attr_name!r}."
            )

    if not hasattr(spec.simulation, "compartments"):
        raise CompileError(
            "Model spec simulation is missing required attribute 'compartments'."
        )

    if not hasattr(spec.simulation, "initializer"):
        raise CompileError(
            "Model spec simulation is missing required attribute 'initializer'."
        )


def run_spec_validation_hooks(
    spec: Any,
    *,
    options: CompileOptions,
) -> None:
    """
    Run optional validation hooks exposed by specs.

    Most validation should already have happened during Pydantic construction.
    These hooks are useful when a nested spec wants to validate itself against
    the full model.
    """
    validate_cross_links = getattr(spec, "validate_cross_links", None)

    if callable(validate_cross_links):
        validate_cross_links()

    initializer = spec.simulation.initializer

    validate_initializer_against_model = getattr(
        initializer,
        "validate_against_model",
        None,
    )

    if callable(validate_initializer_against_model):
        validate_initializer_against_model(spec)
    else:
        validate_initializer_against_simulation = getattr(
            initializer,
            "validate_against_simulation",
            None,
        )

        if callable(validate_initializer_against_simulation):
            validate_initializer_against_simulation(spec.simulation)

    data = getattr(spec, "data", None)

    if data is not None and options.validate_data_spec:
        validate_data_against_model = getattr(
            data,
            "validate_against_model",
            None,
        )

        if callable(validate_data_against_model):
            validate_data_against_model(spec)
        else:
            validate_data_against_simulation = getattr(
                data,
                "validate_against_simulation",
                None,
            )

            if callable(validate_data_against_simulation):
                validate_data_against_simulation(spec.simulation)

    deterministic_execution_order = getattr(
        spec.parameters,
        "deterministic_execution_order",
        None,
    )

    if callable(deterministic_execution_order):
        deterministic_execution_order()

    interaction_matrix_spec = getattr(
        spec.transmission,
        "interaction_matrix_spec",
        None,
    )

    if callable(interaction_matrix_spec):
        interaction_matrix_spec()