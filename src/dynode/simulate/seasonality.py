from datetime import date

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike


def seasonality(
    t: ArrayLike,
    seasonality_amplitude: Array,
    seasonality_second_wave: Array,
    seasonality_dominant_wave_day: Array,
    seasonality_second_wave_day: Array,
    seasonality_wave_width: Array,
    init_date: date,
) -> ArrayLike:
    """Computes a multiplier value for seasonal forcing."""

    def g(t: ArrayLike, phi, w):
        return jnp.abs(jnp.cos(2 * jnp.pi * (t - phi) / 730.0)) ** w

    def f(
        t: ArrayLike,
        dominant_wave_day,
        second_wave_day,
        seasonality_second_wave,
        w,
    ):
        # w determines the narrowness of the waves
        return (
            -0.5
            + 1.0 * g(t, phi=dominant_wave_day, w=w)
            + seasonality_second_wave * g(t, phi=second_wave_day, w=w)
        )

    shifted_dominant_wave_day = (
        seasonality_dominant_wave_day - init_date.timetuple().tm_yday
    )
    shifted_second_wave_day = (
        seasonality_second_wave_day - init_date.timetuple().tm_yday
    )

    return (
        1
        + f(
            t,
            dominant_wave_day=shifted_dominant_wave_day,
            second_wave_day=shifted_second_wave_day,
            seasonality_second_wave=seasonality_second_wave,
            w=seasonality_wave_width,
        )
        * seasonality_amplitude
        * 2
    )
