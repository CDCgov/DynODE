from __future__ import annotations

import json
import os
from datetime import date, datetime
from typing import Any

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.scipy.stats.norm import pdf
from jaxtyping import ArrayLike


def seasonality_coswave(
    t: ArrayLike,
    amplitude: Array,
    peak_day_of_year: Array,
    init_date: date,
) -> ArrayLike:
    """Cosine seasonal forcing multiplier."""

    shifted_t = t + init_date.timetuple().tm_yday
    return (
        1
        + jnp.cos(2 * jnp.pi * (shifted_t - peak_day_of_year) / 365.0)
        * amplitude
    )


def external_i(
    t: ArrayLike,
    introduction_times: Array,
    introduction_scales: Array,
    introduction_percentages: Array,
    introduction_ages: Array,
    population_by_age: Array,
    i_shape: tuple[int, ...],
) -> Array:
    """External infected individuals for strain introductions."""

    densities = pdf(t, loc=introduction_times, scale=introduction_scales)
    introduction_densities = introduction_percentages * densities
    populations = introduction_ages * population_by_age
    populations_new_infections = introduction_densities[:, None] * populations

    ext_i = jnp.zeros(i_shape)
    return ext_i.at[:, 0, 0, 0, 0, :].set(populations_new_infections.T)


def load_vaccination_model_hill(
    region_name: str,
    hill_data_path: str,
    num_age_groups: int,
    init_date: date,
) -> dict[str, Any]:
    """Load age-specific Hill vaccination parameters from JSON input."""

    vax_hill_filename = (
        f"{region_name.lower().replace(' ', '_')}_hillvax_config.json"
    )
    vax_hill_path = os.path.join(hill_data_path, vax_hill_filename)

    if not os.path.exists(vax_hill_path):
        raise FileNotFoundError(
            f"Unable to locate vaccination input data file: {vax_hill_path}"
        )

    vax_input_file = json.load(open(vax_hill_path))

    age_bin_translator = {
        "0-4": 0,
        "5-17": 1,
        "18-49": 2,
        "50-64": 3,
        "65+": 4,
    }

    parameters_to_extract = ("shape", "t_halfsat", "scale", "start_date")
    vax_data = {
        parameter: np.zeros(num_age_groups)
        for parameter in parameters_to_extract
    }

    for age_specific_hill_equation in vax_input_file:
        age_idx = age_bin_translator[age_specific_hill_equation["ageclass"]]

        for parameter in parameters_to_extract:
            if parameter == "start_date":
                start_date = datetime.strptime(
                    age_specific_hill_equation[parameter], "%Y-%m-%d"
                ).date()
                vax_data[parameter][age_idx] = (start_date - init_date).days
            else:
                vax_data[parameter][age_idx] = age_specific_hill_equation[
                    parameter
                ]

    return {
        "ns": jnp.asarray(vax_data["shape"]),
        "t_hs": jnp.asarray(vax_data["t_halfsat"]),
        "t_shifts": jnp.asarray(vax_data["start_date"]),
        "scales": jnp.asarray(vax_data["scale"]),
    }


def vaccination_rate_hill(
    t: ArrayLike,
    t_shifts: Array,
    ns: Array,
    t_hs: Array,
    scales: Array,
    multipliers: Array,
) -> Array:
    """Age-specific vaccination rates based on a Hill equation."""

    shift_coefs = jnp.heaviside(t - t_shifts, 0)
    t_shifted = jnp.where(t - t_shifts < 0, 0, t - t_shifts)
    hill_equation_eval = (
        scales
        * ((ns * (t_shifted ** (ns - 1))) * (t_hs**ns))
        / ((t_shifted**ns + t_hs**ns) ** 2)
    )
    return multipliers * jnp.nan_to_num(shift_coefs * hill_equation_eval)


def beta_modifier_constant(t: ArrayLike, constant: float = 1.0):
    return jnp.full_like(t, constant)


def beta_modifier_step(t: ArrayLike, change_points: Array, multipliers: Array):
    ind = jnp.searchsorted(change_points, t, side="right")
    return multipliers[ind - 1]


def beta_modifier_step_covid(
    t: ArrayLike,
    change_points: Array,
    multipliers: Array,
    covid_change_points: Array,
    covid_multipliers: Array,
):
    return beta_modifier_step(
        t, change_points, multipliers
    ) * beta_modifier_step(t, covid_change_points, covid_multipliers)
