from __future__ import annotations

from dynode.structure.dimensions.immune_history import (
    FullStratifiedImmuneHistoryDimension,
    ImmuneHistoryDimension,
    LastStrainImmuneHistoryDimension,
)

from .references import walk_references


def validate_immune_history_dimensions_match_strains(model) -> None:
    immune_dims = [
        dim
        for dim in model.simulation.flatten_dims()
        if isinstance(dim, ImmuneHistoryDimension)
    ]

    for dim in immune_dims:
        if not isinstance(
            dim,
            (
                FullStratifiedImmuneHistoryDimension,
                LastStrainImmuneHistoryDimension,
            ),
        ):
            raise ValueError(
                f"Unsupported immune-history dimension type: {type(dim).__name__}."
            )

        dim.validate_against_strains(model.strains)


def validate_introduced_strain_ages(model) -> None:
    strains_with_intro_ages = [
        strain
        for strain in model.strains
        if strain.is_introduced and strain.introduction_ages is not None
    ]

    if not strains_with_intro_ages:
        return

    age_bins = model.age_bins

    if not age_bins:
        names = [strain.name for strain in strains_with_intro_ages]
        raise ValueError(
            "Some introduced strains define introduction_ages, but the "
            f"simulation has no age dimension. Strains: {names}."
        )

    for strain in strains_with_intro_ages:
        missing = [
            age for age in strain.introduction_ages if age not in age_bins
        ]

        if missing:
            raise ValueError(
                f"Strain {strain.name!r} defines introduction_ages that are not "
                f"present in the simulation age bins. Missing bins: {missing}."
            )


def validate_model_references(model) -> None:
    available = model.available_parameter_names
    deterministic_names = set(
        model.parameters.deterministic_parameter_names
    ) | set(model.external_parameter_names)

    references = list(
        walk_references(model.transmission, path="transmission")
    )
    references += list(
        walk_references(
            model.simulation.initializer, path="simulation.initializer"
        )
    )

    unknown_parameter_refs: list[str] = []
    unknown_deterministic_refs: list[str] = []

    for path, kind, name in references:
        if kind == "parameter" and name not in available:
            unknown_parameter_refs.append(f"{path} -> {name!r}")

        if kind == "deterministic" and name not in deterministic_names:
            unknown_deterministic_refs.append(f"{path} -> {name!r}")

    errors: list[str] = []

    if unknown_parameter_refs:
        errors.append(
            "Unknown parameter references: "
            + ", ".join(unknown_parameter_refs)
        )

    if unknown_deterministic_refs:
        errors.append(
            "Unknown deterministic references: "
            + ", ".join(unknown_deterministic_refs)
        )

    if errors:
        raise ValueError("; ".join(errors))


def validate_initializer_against_model(model) -> None:
    validate_against_model = getattr(
        model.simulation.initializer, "validate_against_model", None
    )

    if callable(validate_against_model):
        validate_against_model(model)


def validate_data_against_model(model) -> None:
    if model.data is None:
        return

    validate_against_model = getattr(
        model.data, "validate_against_model", None
    )

    if callable(validate_against_model):
        validate_against_model(model)
