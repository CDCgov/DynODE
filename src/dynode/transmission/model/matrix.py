from __future__ import annotations

from typing import Any

from dynode.transmission.interactions.spec import InteractionSpec


def interaction_value(
    *,
    transmission: Any,
    source_name: str,
    target_name: str,
) -> InteractionSpec:
    """Return the interaction spec for source -> target."""
    source = transmission.get_strain(source_name)

    if target_name in source.interactions:
        return source.interactions[target_name]

    if source_name == target_name and transmission.force_diag_ones:
        return InteractionSpec.fixed(1.0)

    return transmission.default_offdiag


def interaction_matrix_spec(transmission: Any) -> list[list[InteractionSpec]]:
    names = transmission.strain_names

    return [
        [
            interaction_value(
                transmission=transmission,
                source_name=source_name,
                target_name=target_name,
            )
            for target_name in names
        ]
        for source_name in names
    ]


def interaction_matrix_named(
    transmission: Any,
) -> dict[str, dict[str, InteractionSpec]]:
    names = transmission.strain_names

    return {
        source_name: {
            target_name: interaction_value(
                transmission=transmission,
                source_name=source_name,
                target_name=target_name,
            )
            for target_name in names
        }
        for source_name in names
    }


def interaction_matrix_dump(transmission: Any) -> list[list[dict[str, Any]]]:
    return [
        [value.model_dump() for value in row]
        for row in interaction_matrix_spec(transmission)
    ]
