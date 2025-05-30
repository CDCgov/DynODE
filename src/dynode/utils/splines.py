"""Defines functions for evaluating cubic splines."""

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike


# Vaccination modeling, using cubic splines to model vax uptake
# in the population stratified by age and current vax shot.
def base_equation(t: ArrayLike, coefficients: Array) -> Array:
    """Compute the base of a spline equation without knots.

    Follows a simple cubic formula: a + bt + ct^2 + dt^3.
    This is a vectorized version that takes in a matrix of
    coefficients for each age x vaccination combination.

    Parameters
    ----------
    t : jax.ArrayLike
        Simulation day.
    coefficients : jnp.ndarray
        Coefficients of each cubic spline base equation for all
        combinations of age bin and vaccination history.
        Shape: (NUM_AGE_GROUPS, MAX_VACCINATION_COUNT + 1, 4)

    Returns
    -------
    jnp.ndarray
        The result of executing the base equation `a + bt + ct^2 + dt^3`
        for each age group and vaccination count combination.
        Shape: (NUM_AGE_GROUPS, MAX_VACCINATION_COUNT + 1)
    """
    return jnp.sum(
        coefficients
        * jnp.array([1, t, t**2, t**3])[jnp.newaxis, jnp.newaxis, :],
        axis=-1,
    )


def conditional_knots(
    t: ArrayLike, knots: Array, coefficients: Array
) -> Array:
    """Evaluate knots of a spline.

    Evaluates combination of an indicator variable and the
    coefficient associated with that knot.

    Executes the following equation:
    sum_{i}^{len(knots)}(coefficients[i] * (t - knots[i])^3 * I(t > knots[i]))
    where I() is an indicator variable.

    Parameters
    ----------
    t : jax.ArrayLike
        Simulation day.
    knots : jax.Array
        Knot locations to compare with `t`.
    coefficients : jax.Array
        Knot coefficients to multiply each knot with,
        assuming it is active at some timestep `t`.

    Returns
    -------
    jax.Array
        Resulting values summed over the last dimension of the matrices.
    """
    indicators = jnp.where(t > knots, t - knots, 0)
    # multiply coefficients by 3 since we taking derivative of cubic spline.
    return jnp.sum(indicators**3 * coefficients, axis=-1)


def evaluate_cubic_spline(
    t,
    knot_locations: Array,
    base_equations: Array,
    knot_coefficients: Array,
) -> Array:
    """Evaluate a cubic spline with knots and coefficients on day `t`.

    Cubic spline equation age_bin x vaccination history combination:
    ```
    f(t) = a + bt + ct^2 + dt^3 +
        sum_{i}^{len(knot_locations)}(knot_coefficients[i]
        * (t - knot_locations[i])^3
        * I(t > knot_locations[i]))
    ```

    Parameters
    ----------
    t : jax.ArrayLike
        Simulation day.
    knot_locations : jnp.ndarray
        Knot locations for all combinations of age bin and vaccination history.
        Shape: (NUM_AGE_GROUPS, MAX_VACCINATION_COUNT + 1, #knots)
    base_equations : jnp.ndarray
        Base equation coefficients (a + bt + ct^2 + dt^3) for all combinations of age bin and vaccination history.
        Shape: (NUM_AGE_GROUPS, MAX_VACCINATION_COUNT + 1, 4)
    knot_coefficients : jnp.ndarray
        Knot coefficients for all combinations of age bin and vaccination history.
        Shape: (NUM_AGE_GROUPS, MAX_VACCINATION_COUNT + 1, #knots)

    Returns
    -------
    jnp.ndarray
        Proportion of individuals in each age x vaccination combination vaccinated during this time step.
    """
    base = base_equation(t, base_equations)
    knots = conditional_knots(t, knot_locations, knot_coefficients)
    return base + knots
