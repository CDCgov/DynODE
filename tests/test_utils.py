import pytest
import utils
import jax.numpy as jnp
from enum import IntEnum


# strain indexes {"a": 0, "b": 1, "c": 2}
example_strain_idxs = IntEnum("test", ["a", "b", "c"], start=0)
num_strains = 3


def test_convert_strain():
    assert utils.convert_strain("a", example_strain_idxs) == 0
    assert utils.convert_strain("b", example_strain_idxs) == 1
    assert utils.convert_strain("c", example_strain_idxs) == 2
    assert utils.convert_strain("C", example_strain_idxs) == 2
    assert utils.convert_strain("not_in_idxs", example_strain_idxs) == 0


def test_base_equation():
    # attempt to test: 5 + 1t + 2t^2 + 3t^3
    def equation(t):
        return 5 + t + (2 * t**2) + (3 * t**3)

    coefficients = jnp.array([5, 1, 2, 3])
    tested_times = list(range(-2, 15)) + [100]
    for time in tested_times:
        assert utils.base_equation(time, coefficients) == equation(
            time
        ), "base equation failed to evaluate with input : %s" % str(time)


def test_conditional_knots_no_coefficients():
    # attempt to test a cubic spline with knots at t=0, 5, 10
    # equation as follows:
    # f(t) = ((t-0) ^ 3 * I(t > 0)) + ((t-5) ^ 3 * I(t > 5)) + ((t-10) ^ 3 * I(t > 10))
    def equation(t):
        return (
            ((t - 0) ** 3 * (t > 0))
            + ((t - 5) ** 3 * (t > 5))
            + ((t - 10) ** 3 * (t > 10))
        )

    # test with no coefficients first
    coefficients = jnp.array([1, 1, 1])
    knots = jnp.array([0, 5, 10])
    tested_times = list(range(-2, 15)) + [100]
    for time in tested_times:
        assert utils.conditional_knots(time, knots, coefficients) == equation(
            time
        ), "conditional_knots failed to evaluate with input : %s" % str(time)


def test_conditional_knots_with_coefficients():
    # attempt to test a cubic spline with knots at t=0, 5, 10
    # equation as follows:
    # f(t) = ((t-0) ^ 3 * I(t > 0)) + ((t-5) ^ 3 * I(t > 5)) + ((t-10) ^ 3 * I(t > 10))
    # add 1, 2, 3 as coefficients to each of the knots.
    def equation(t):
        return (
            (1 * (t - 0) ** 3 * (t > 0))
            + (2 * (t - 5) ** 3 * (t > 5))
            + (3 * (t - 10) ** 3 * (t > 10))
        )

    knots = jnp.array([0, 5, 10])
    coefficients = jnp.array([1, 2, 3])
    tested_times = list(range(-2, 15)) + [100]
    for time in tested_times:
        assert utils.conditional_knots(time, knots, coefficients) == equation(
            time
        ), "conditional_knots failed to evaluate with input : %s" % str(time)


def test_cubic_spline():
    # test the following equations
    # f(t): 1 + 2t + 3t**2 + 4t**3 + (5(t-0)**3 * I(t > 0)) + (6(t-0)**3 * I(t > 5)) + (7(t-0)**3 * I(t > 10))
    def equation(t):
        return (
            1
            + 2 * t
            + (3 * t**2)
            + (4 * t**3)
            + (5 * (t - 0) ** 3 * (t > 0))
            + (6 * (t - 5) ** 3 * (t > 5))
            + (7 * (t - 10) ** 3 * (t > 10))
        )

    knot_locations = jnp.array([0, 5, 10])
    base_equation_coefficients = jnp.array([1, 2, 3, 4])
    knot_coefficients = jnp.array([5, 6, 7])
    tested_times = list(range(-2, 15)) + [100]
    for time in tested_times:
        assert utils.evaluate_cubic_spline(
            time,
            knot_locations,
            base_equation_coefficients,
            knot_coefficients,
        ) == equation(
            time
        ), "evaluate_cubic_spline failed to evaluate with input : %s" % str(
            time
        )


def test_new_immune_state():
    num_strains_tested = [1, 2, 3, 4, 10]
    for num_strains in num_strains_tested:
        num_possible_immune_states = list(range(0, 2**num_strains))
    utils.new_immune_state()
