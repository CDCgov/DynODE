from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np

from dynode.value.constant import ConstantValueSpec


def validate_core_values(strain: Any) -> None:
    validate_constant_non_negative(strain.r0, field_name="r0")
    validate_constant_positive(
        strain.infectious_period,
        field_name="infectious_period",
    )

    if strain.exposed_to_infectious is not None:
        validate_constant_positive(
            strain.exposed_to_infectious,
            field_name="exposed_to_infectious",
        )


def validate_introduction_fields(strain: Any) -> None:
    introduction_fields = {
        "introduction_time": strain.introduction_time,
        "introduction_percentage": strain.introduction_percentage,
        "introduction_scale": strain.introduction_scale,
        "introduction_ages": strain.introduction_ages,
    }

    provided = [
        field_name
        for field_name, value in introduction_fields.items()
        if value is not None
    ]

    if not strain.is_introduced:
        if provided:
            raise ValueError(
                "Introduction fields were provided, but is_introduced=False. "
                f"Provided fields: {provided}."
            )

        return

    required = (
        "introduction_time",
        "introduction_percentage",
        "introduction_scale",
    )

    missing = [
        field_name
        for field_name in required
        if getattr(strain, field_name) is None
    ]

    if missing:
        raise ValueError(
            "Introduced strains must define introduction_time, "
            "introduction_percentage, and introduction_scale. "
            f"Missing: {missing}."
        )

    validate_constant_non_negative_or_date(
        strain.introduction_time,
        field_name="introduction_time",
    )

    validate_constant_positive(
        strain.introduction_percentage,
        field_name="introduction_percentage",
    )

    validate_constant_positive(
        strain.introduction_scale,
        field_name="introduction_scale",
    )


def validate_no_data_dependencies(
    *,
    strain_name: str,
    data_dependencies: set[str],
) -> None:
    if data_dependencies:
        raise ValueError(
            f"Strain {strain_name!r} contains data references, which are not "
            f"allowed in StrainSpec. Data dependencies: {sorted(data_dependencies)}."
        )


def validate_constant_non_negative(value: Any, field_name: str) -> None:
    if not isinstance(value, ConstantValueSpec):
        return

    array = constant_numeric_array(value, field_name)

    if np.any(array < 0):
        raise ValueError(f"{field_name} must be non-negative.")


def validate_constant_positive(value: Any, field_name: str) -> None:
    if not isinstance(value, ConstantValueSpec):
        return

    array = constant_numeric_array(value, field_name)

    if np.any(array <= 0):
        raise ValueError(f"{field_name} must be positive.")


def validate_constant_non_negative_or_date(
    value: Any,
    field_name: str,
) -> None:
    if not isinstance(value, ConstantValueSpec):
        return

    raw = value.value

    if isinstance(raw, date):
        return

    array = constant_numeric_array(value, field_name)

    if np.any(array < 0):
        raise ValueError(
            f"{field_name} must be a date or non-negative simulation time."
        )


def constant_numeric_array(
    value: ConstantValueSpec,
    field_name: str,
) -> np.ndarray:
    raw = value.value

    if isinstance(raw, bool):
        raise ValueError(f"{field_name} must be numeric, not bool.")

    if isinstance(raw, date):
        raise ValueError(f"{field_name} must be numeric, not date.")

    try:
        return np.asarray(raw, dtype=float)
    except Exception as exc:
        raise ValueError(f"{field_name} must be numeric.") from exc
