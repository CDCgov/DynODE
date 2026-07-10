from __future__ import annotations

from dynode.structure.dimensions.base import DimensionSpec


def validate_unique_compartment_names(simulation) -> None:
    names = simulation.compartment_names
    duplicates = sorted({name for name in names if names.count(name) > 1})

    if duplicates:
        raise ValueError(
            f"Compartment names must be unique. Duplicates: {duplicates}."
        )


def validate_unique_dimension_names_within_compartments(simulation) -> None:
    for compartment in simulation.compartments:
        dimension_names = [
            dimension.name for dimension in compartment.dimensions
        ]
        duplicates = sorted(
            {
                name
                for name in dimension_names
                if dimension_names.count(name) > 1
            }
        )

        if duplicates:
            raise ValueError(
                f"Compartment {compartment.name!r} contains duplicate "
                f"dimension names: {duplicates}."
            )


def validate_shared_dimensions_are_identical(simulation) -> None:
    dimension_by_name: dict[str, DimensionSpec] = {}

    for dimension in simulation.flatten_dims():
        previous = dimension_by_name.get(dimension.name)

        if previous is None:
            dimension_by_name[dimension.name] = dimension
            continue

        if dimension != previous:
            raise ValueError(
                f"Dimension {dimension.name!r} has inconsistent definitions "
                "across compartments. If these are intended to be different, "
                "give them different names."
            )


def validate_initializer_compatible(simulation) -> None:
    validate = getattr(
        simulation.initializer,
        "validate_against_simulation",
        None,
    )

    if callable(validate):
        validate(simulation)
