from datetime import date

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from .splines import evaluate_cubic_spline


def vaccination_rate_spline(
    t: ArrayLike,
    vaccination_model_knot_locations: Array,
    vaccination_model_base_equations: Array,
    vaccination_model_knots: Array,
) -> Array:
    """Computes rates of vaccination using a spline model."""
    vaccination_rates_log = evaluate_cubic_spline(
        t,
        vaccination_model_knot_locations,
        vaccination_model_base_equations,
        vaccination_model_knots,
    )
    dummy = jnp.zeros(vaccination_rates_log.shape)
    # print(jax.dtypes(dummy))
    # one of the side effects of exp() is setting exp(0) -> 1
    # we dont want this behavior in our vaccination rates obviously
    # so we find the locations of zero and save them to remask 0 -> 0 after exp() op
    zero_mask = jnp.where(vaccination_rates_log == 0, 0, 1)
    return zero_mask * jnp.exp(vaccination_rates_log + dummy)


def vaccination_rate_hill(
    t,
    t_shifts: Array,
    n_batched: Array,
    t_h_batched: Array,
    scales_batched: Array,
) -> Array:
    """Compute vaccination rates based on a hill equation."""

    def hill_equation_dt(t, n, t_h, scale):
        """Derivative of the hill equation with respect to `t`, used
        to calculate uptake at day=t with shape parameters `n`, time of
        half cumulative uptake `t_h`, target end saturation date `t_end`
        and ending cumulative vaccination uptake `y_t_end`"""
        t1 = t[:, jnp.newaxis]
        return (
            scale
            * ((n * (t1 ** (n - 1))) * (t_h**n))
            / ((t1**n + t_h**n) ** 2)
        )

    shift_coefs = jnp.heaviside(t - t_shifts, 0)
    t_shift = jnp.where(t - t_shifts < 0, 0, t - t_shifts)
    return jnp.nan_to_num(
        shift_coefs[:, jnp.newaxis]
        * hill_equation_dt(
            t_shift,
            n_batched,
            t_h_batched,
            scales_batched,
        )
    )


def seasonal_vaccine_reset(
    t: ArrayLike, vaccine_season_change: date, init_date: date
) -> ArrayLike:
    """Computes a multiplier value for resetting seasonal vaccinations in the summer."""
    # outflow function must be positive if and only if
    # it is time to move people from seasonal bin back to max ordinal bin
    # use a sine wave that occurs once a year to achieve this effect
    peak_of_function = 182.5
    # shift this value using shift_t to align with config.VACCINATION_SEASON_CHANGE
    # such that outflow_fn(config.VACCINATION_SEASON_CHANGE) == 1.0 always
    shift_t = peak_of_function - (vaccine_season_change - init_date).days
    # raise to an even exponent to remove negatives,
    # pick 1000 since too high of a value likely to be stepped over by adaptive step size
    # divide by 730 so wave only occurs 1 per every 365 days
    # multiply by 2pi since we pass days as int
    return jnp.sin((2 * jnp.pi * (t + shift_t) / 730)) ** 1000
