from __future__ import annotations

from typing import Any


def validate_unique_strain_names(transmission: Any) -> None:
    names = transmission.strain_names
    duplicates = sorted({name for name in names if names.count(name) > 1})

    if duplicates:
        raise ValueError(f"Duplicate strain names found: {duplicates}")


def validate_interaction_targets(transmission: Any) -> None:
    valid_names = set(transmission.strain_names)

    for strain in transmission.strains:
        unknown_targets = set(strain.interactions) - valid_names

        if unknown_targets:
            raise ValueError(
                f"Strain {strain.name!r} defines interactions for unknown "
                f"target strains: {sorted(unknown_targets)}. "
                f"Known strains are: {sorted(valid_names)}."
            )


def validate_introduction_ages_consistent(transmission: Any) -> None:
    introduced = [
        strain
        for strain in transmission.strains
        if getattr(strain, "is_introduced", False)
    ]

    intro_age_sets = [
        tuple(strain.introduction_ages)
        for strain in introduced
        if getattr(strain, "introduction_ages", None) is not None
    ]

    if intro_age_sets:
        first = intro_age_sets[0]
        mismatched = [
            strain.name
            for strain in introduced
            if getattr(strain, "introduction_ages", None) is not None
            and tuple(strain.introduction_ages) != first
        ]

        if mismatched:
            raise ValueError(
                "Currently all introduced strains must have matching "
                f"introduction_ages. Mismatched strains: {mismatched}"
            )


def validate_optional_strain_fields_consistent(transmission: Any) -> None:
    """Require selected optional strain-level fields to be all-or-none."""
    optional_fields_to_check = [
        "exposed_to_infectious",
        "vaccine_efficacy",
    ]

    for field_name in optional_fields_to_check:
        present = [
            strain.name
            for strain in transmission.strains
            if getattr(strain, field_name, None) is not None
        ]

        if present and len(present) != len(transmission.strains):
            missing = [
                strain.name
                for strain in transmission.strains
                if getattr(strain, field_name, None) is None
            ]

            raise ValueError(
                f"If {field_name!r} is set for one strain, it must be set "
                f"for all strains. Present={present}, missing={missing}."
            )
