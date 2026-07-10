from __future__ import annotations

from typing import Any


def evaluated_strain_values(
    *,
    strain: Any,
    context: dict[str, Any] | None = None,
    data: Any | None = None,
) -> dict[str, Any]:
    """Evaluate strain-level values against a runtime context."""
    values: dict[str, Any] = {
        "name": strain.name,
        "r0": strain.r0.evaluate(context=context, data=data),
        "infectious_period": strain.infectious_period.evaluate(
            context=context,
            data=data,
        ),
        "is_introduced": strain.is_introduced,
    }

    if strain.exposed_to_infectious is not None:
        values["exposed_to_infectious"] = (
            strain.exposed_to_infectious.evaluate(
                context=context,
                data=data,
            )
        )

    if strain.vaccine_efficacy is not None:
        values["vaccine_efficacy"] = dict(strain.vaccine_efficacy)

    if strain.is_introduced:
        values["introduction_time"] = strain.introduction_time.evaluate(
            context=context,
            data=data,
        )
        values["introduction_percentage"] = (
            strain.introduction_percentage.evaluate(
                context=context,
                data=data,
            )
        )
        values["introduction_scale"] = strain.introduction_scale.evaluate(
            context=context,
            data=data,
        )
        values["introduction_ages"] = strain.introduction_ages

    return values
