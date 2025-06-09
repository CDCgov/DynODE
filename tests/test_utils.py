import datetime
from enum import IntEnum

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist

from dynode import utils

# strain indexes {"a": 0, "b": 1, "c": 2}

example_strain_idxs = IntEnum("example_strain_idxs", ["a", "b", "c"], start=0)
num_strains = 3


def test_base_equation():
    # attempt to test: 5 + 1t + 2t^2 + 3t^3
    def equation(t):
        return 5 + t + (2 * t**2) + (3 * t**3)

    coefficients = jnp.array([5, 1, 2, 3])
    tested_times = list(range(-2, 15)) + [100]
    for time in tested_times:
        assert utils.base_equation(time, coefficients) == equation(time), (
            "base equation failed to evaluate with input : %s" % str(time)
        )


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
        ) == equation(time), (
            "evaluate_cubic_spline failed to evaluate with input : %s"
            % str(time)
        )


def test_evaluate_cubic_spline():
    test_base_equations = jnp.array([[1, 2, 3, 4], [1, 2, 3, 4]])
    test_spline_locations = jnp.array([[0, 2, 4, 6], [0, 2, 4, 6]])
    test_spline_coefs = jnp.array([[1, 1, 1, 1], [1, 2, 3, -4]])

    def test_spline_1(t):
        base_equation = 1 + 2 * t + 3 * t**2 + 4 * t**3
        # coefficients all 1, just sum the indicators
        spline_indicators = t > test_spline_locations[0]
        splines = jnp.sum(
            (t - test_spline_locations[0]) ** 3 * spline_indicators
        )
        return base_equation + splines

    def test_spline_2(t):
        base_equation = 1 + 2 * t + 3 * t**2 + 4 * t**3
        # coefficients all 1, just sum the indicators
        spline_indicators = t > test_spline_locations[1]  # indicator vars
        splines = jnp.sum(
            test_spline_coefs[1]
            * ((t - test_spline_locations[1]) ** 3 * spline_indicators)
        )
        return base_equation + splines

    for t in range(-5, 5, 1):
        utils_splines = utils.evaluate_cubic_spline(
            t, test_spline_locations, test_base_equations, test_spline_coefs
        ).flatten()
        assert utils_splines[0] == test_spline_1(t), (
            "utils.evaluate_cubic_spline is returning incorrect splines, check the math at t=%d"
            % t
        )
        assert utils_splines[1] == test_spline_2(t), (
            "utils.evaluate_cubic_spline is returning incorrect splines, check the math at t=%d"
            % t
        )


def test_date_to_epi_week():
    random_date_looked_up_epi_week_for = datetime.date(2024, 2, 1)
    epi_week_found_on_cdc_calendar = 5
    epi_week_returned = utils.date_to_epi_week(
        random_date_looked_up_epi_week_for
    ).week
    assert epi_week_returned == epi_week_found_on_cdc_calendar, (
        "date_to_epi_week returns incorrect epi week for feb 1st 2024, got %s, should be %s"
        % (epi_week_returned, epi_week_found_on_cdc_calendar)
    )


def test_identify_distribution_indexes():
    parameters = {
        "test": [0, dist.Normal(), 2],
        "example": dist.Normal(),
        "no-sample": 5,
    }
    indexes = utils.identify_distribution_indexes(parameters)

    assert "test_1" in indexes.keys() and indexes["test_1"] == {
        "sample_name": "test",
        "sample_idx": tuple([1]),
    }, "not correctly indexing sampled parameters within lists"
    assert "example" in indexes.keys() and indexes["example"] == {
        "sample_name": "example",
        "sample_idx": None,
    }, "not correctly indexing non-list sampled parameters"
    assert "no-sample" not in indexes.keys(), (
        "identify_distribution_indexes should not return indexes for unsampled parameters"
    )


# get the function to test
def _get_index_enums():
    compartment_idx = IntEnum(
        "compartment_index", ["S", "E", "I", "C"], start=0
    )
    wane_idx = IntEnum("wane_index", ["W0", "W1", "W2", "W3"], start=0)
    strain_idx = IntEnum("strain_index", ["S0", "S1", "S2", "S3"], start=0)
    return compartment_idx, wane_idx, strain_idx


def _get_sol():
    return tuple(
        [
            jnp.ones(
                (100, 4, 4, 4, 4),
            )
            for _ in range(4)
        ]
    )


def test_flatten_list_params_numpy():
    # simulate 4 chains and 20 samples each with 4 plated parameters
    testing = {"test": np.ones((4, 20, 5))}
    flattened = utils.flatten_list_parameters(testing)
    assert "test" not in flattened.keys()
    for suffix in range(5):
        key = "test_%s" % str(suffix)
        assert key in flattened.keys(), (
            "flatten_list_parameters not naming split params correctly."
        )
        assert flattened[key].shape == (
            4,
            20,
        ), "flatten_list_parameters breaking up wrong axis"


def test_flatten_list_params_jax_numpy():
    # simulate 4 chains and 20 samples each with 4 plated parameters
    # this time with jax numpy instead of numpy
    testing = {"test": jnp.ones((4, 20, 5))}
    flattened = utils.flatten_list_parameters(testing)
    assert "test" not in flattened.keys()
    for suffix in range(5):
        key = "test_%s" % str(suffix)
        assert key in flattened.keys(), (
            "flatten_list_parameters not naming split params correctly."
        )
        assert flattened[key].shape == (
            4,
            20,
        ), "flatten_list_parameters breaking up wrong axis"


def test_flatten_list_params_multi_dim():
    # simulate 4 chains and 20 samples each with 10 plated parameters
    # this time with jax numpy instead of numpy
    testing = {"test": jnp.ones((4, 20, 5, 2))}
    flattened = utils.flatten_list_parameters(testing)
    assert "test" not in flattened.keys()
    for suffix_first_dim in range(5):
        for suffix_second_dim in range(2):
            key = "test_%s_%s" % (
                str(suffix_first_dim),
                str(suffix_second_dim),
            )
            assert key in flattened.keys(), (
                "flatten_list_parameters not naming split params correctly."
            )
            assert flattened[key].shape == (
                4,
                20,
            ), "flatten_list_parameters breaking up wrong axis when passed >3"


def test_vectorize_objects():
    class TestObject:
        def __init__(self, value):
            self.value = value

    objs = [TestObject(i) for i in range(5)]
    target = "value"

    # Test without filter
    result = utils.vectorize_objects(objs, target)
    assert result == [0, 1, 2, 3, 4], "vectorize_objects failed without filter"

    # Test with filter
    result = utils.vectorize_objects(
        objs, target, filter=lambda x: x.value > 2
    )
    assert result == [3, 4], "vectorize_objects failed with filter"


def test_vectorize_objects_exceptions():
    class TestObject:
        def __init__(self, value):
            self.value = value

    objs = [TestObject(i) for i in range(5)]
    # Test with non-existing attribute
    try:
        utils.vectorize_objects(objs, "non_existing")
    except AttributeError:
        pass  # Expected behavior
    else:
        assert False, (
            "vectorize_objects should raise AttributeError for non-existing attribute"
        )


def test_sim_day_to_date():
    init_date = datetime.date(2022, 10, 15)
    sim_day = 21
    expected_date = init_date + datetime.timedelta(days=sim_day)

    result_date = utils.sim_day_to_date(sim_day, init_date)

    assert result_date == expected_date, (
        f"sim_day_to_date failed: expected {expected_date}, got {result_date}"
    )


def test_sim_day_to_epiweek():
    # this init_date is on a saturday, which means it still belongs
    # to the previous year's epi-weeks, thus 52
    init_date = datetime.date(2022, 1, 1)
    sim_day = 0
    epi_week = utils.sim_day_to_epiweek(sim_day, init_date)

    assert 52 == epi_week.week, (
        f"sim_day_to_epiweek failed: expected 52, got {epi_week.week}"
    )
    # now lets check the next week, ensuring epi-week wraps back around to 1
    sim_day = 2
    epi_week = utils.sim_day_to_epiweek(sim_day, init_date)

    assert 1 == epi_week.week, (
        f"sim_day_to_epiweek failed: expected 1, got {epi_week.week}"
    )


def test_date_to_sim_day():
    init_date = datetime.date(2022, 10, 15)
    date = datetime.date(2022, 11, 5)
    expected_sim_day = (date - init_date).days

    result_sim_day = utils.date_to_sim_day(date, init_date)

    assert result_sim_day == expected_sim_day, (
        f"date_to_sim_day failed: expected {expected_sim_day}, got {result_sim_day}"
    )
