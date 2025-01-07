import datetime
import itertools
from enum import IntEnum

import jax.numpy as jnp
import numpyro.distributions as dist

from dynode import utils

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
        possible_immune_states = list(range(0, 2**num_strains))
        exposing_strains = list(range(0, num_strains))
        for old_state, exposing_strain in itertools.product(
            possible_immune_states, exposing_strains
        ):
            new_state = utils.new_immune_state(old_state, exposing_strain)
            # exposing_strain in binary has 1 in the index of exposing strain, with index 0 being right most
            exposing_strain_binary = ["0"] * num_strains
            exposing_strain_binary[exposing_strain] = "1"
            # invert order so index 0 is right most of string.
            exposing_strain_binary = "".join(exposing_strain_binary[::-1])
            # bitwise OR for new state
            expected_new_state = int(format(old_state, "b"), 2) | int(
                exposing_strain_binary, 2
            )

            assert new_state == expected_new_state, (
                "the new immune state when state %d is exposed to strain %d with a max number of strains %d is incorrect"
                % (old_state, exposing_strain, num_strains)
            )


def test_all_immune_states_with():
    num_strains_tested = [1, 2, 3, 4, 10]
    # testing a number of num_strain variables
    for num_strains in num_strains_tested:
        possible_immune_states = list(range(0, 2**num_strains))
        exposing_strains = list(range(0, num_strains))
        # testing each of the strains
        for strain in exposing_strains:
            states_with_strain = utils.all_immune_states_with(
                strain, num_strains
            )
            for immune_state in possible_immune_states:
                state_binary = format(immune_state, "b")
                # prepend some zeros if needed to avoid index errors
                # invert so we can index `strain`, as opposed to `strain` indexes from the end in a list
                state_binary = (
                    "0" * (num_strains - len(state_binary)) + state_binary
                )[::-1]
                # should contain strain, 1 in the `strain` index of the binary
                if immune_state in states_with_strain:
                    assert state_binary[strain] == "1", (
                        "state %d should have an exposure to strain %d but does not when num_strains is %d"
                        % (immune_state, strain, num_strains)
                    )
                else:  # does not contain strain
                    assert state_binary[strain] == "0", (
                        "state %d should NOT have an exposure to strain %d but does when num_strains is %d"
                        % (immune_state, strain, num_strains)
                    )


def test_all_immune_states_without():
    num_strains_tested = [1, 2, 3, 4, 10]
    # testing a number of num_strain variables
    for num_strains in num_strains_tested:
        possible_immune_states = list(range(0, 2**num_strains))
        exposing_strains = list(range(0, num_strains))
        # testing each of the strains
        for strain in exposing_strains:
            states_without_strain = utils.all_immune_states_without(
                strain, num_strains
            )
            for immune_state in possible_immune_states:
                state_binary = format(immune_state, "b")
                # prepend some zeros if needed to avoid index errors
                # invert so we can index `strain`, as opposed to `strain` indexes from the end in a list
                state_binary = (
                    "0" * (num_strains - len(state_binary)) + state_binary
                )[::-1]
                # should contain strain, 1 in the `strain` index of the binary
                if immune_state in states_without_strain:
                    assert state_binary[strain] == "0", (
                        "state %d should NOT have an exposure to strain %d but does when num_strains is %d"
                        % (immune_state, strain, num_strains)
                    )
                else:  # does not contain strain
                    assert state_binary[strain] == "1", (
                        "state %d should have an exposure to strain %d but does not when num_strains is %d"
                        % (immune_state, strain, num_strains)
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
    assert (
        "no-sample" not in indexes.keys()
    ), "identify_distribution_indexes should not return indexes for unsampled parameters"


def test_combined_strain_mapping():
    pass
    # lets test two hardcoded scenarios, since this code can get a bit crazy
    # 2 strain scenario:
    # 0 -> no exposures,
    # 1-> exposure to strain 0,
    # 2 -> exposure to strain 1,
    # 3 -> exposure to both
    # num_strains_tested = [2, 3, 4, 10]
    # # testing a number of num_strain variables
    # for num_strains in num_strains_tested:
    #     possible_immune_states = list(range(0, 2**num_strains))
    #     exposing_strains = list(range(0, num_strains))
    #     for from_strain, to_strain in itertools.product(
    #         exposing_strains, exposing_strains
    #     ):
    #         combined_state_dict, strain_mapping = (
    #             utils.combined_strains_mapping(
    #                 from_strain, to_strain, num_strains
    #             )
    #         )
    #         print(combined_state_dict)
    #         print(strain_mapping)
    #         for old_state, new_state in combined_state_dict.items():
    #             # we assert that we can go back to the old state by re-exposing new_state to the strain that was combined.
    #             assert utils.new_immune_state(
    #                 new_state, from_strain, num_strains
    #             ) == utils.new_immune_state(
    #                 new_state, to_strain, num_strains
    #             ), (
    #                 "after combining strains %d and %d together, exposing %d to %d did not yield state %d as expected"
    #                 % (
    #                     from_strain,
    #                     to_strain,
    #                     new_state,
    #                     from_strain,
    #                     old_state,
    #                 )
    #             )


def test_get_strains_exposed_to():
    num_strains_tested = [1, 2, 3, 4, 10]
    for num_strains in num_strains_tested:
        possible_immune_states = list(range(0, 2**num_strains))
        for state in possible_immune_states:
            exposed_strains = utils.get_strains_exposed_to(state, num_strains)
            # Calculate the expected strains exposed by converting the state to binary
            expected_exposed_strains = [
                i for i in range(num_strains) if (state & (1 << i)) != 0
            ]
            assert exposed_strains == expected_exposed_strains, (
                f"The exposed strains for state {state} with {num_strains} strains is incorrect. "
                f"Expected {expected_exposed_strains}, got {exposed_strains}."
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


def test_get_timeseries_from_solution_with_command_compartment_name():
    # Test case 1: Command is a compartment name
    sol = _get_sol()
    compartment_idx, wane_idx, strain_idx = _get_index_enums()
    timeseries, label = utils.get_timeseries_from_solution_with_command(
        sol, compartment_idx, wane_idx, strain_idx, "S"
    )
    assert timeseries.shape == (100,)
    assert label == "S"
    assert jnp.all(
        timeseries == 256
    )  # Each element in sol is 1, summed over 4*4*4*4 = 256


def test_get_timeseries_from_solution_with_command_strain_name():
    # Test case 2: Command is a strain name
    sol = _get_sol()
    compartment_idx, wane_idx, strain_idx = _get_index_enums()
    timeseries, label = utils.get_timeseries_from_solution_with_command(
        sol, compartment_idx, wane_idx, strain_idx, "S2"
    )
    assert timeseries.shape == (100,)
    assert label == "E + I : S2"
    assert jnp.all(
        timeseries == 128
    )  # Exposed + Infected, both are 64 each and of course 64*2==128


def test_get_timeseries_from_solution_with_command_wane_name():
    # Test case 3: Command is a waning compartment name
    sol = _get_sol()
    compartment_idx, wane_idx, strain_idx = _get_index_enums()
    timeseries, label = utils.get_timeseries_from_solution_with_command(
        sol, compartment_idx, wane_idx, strain_idx, "W0"
    )
    assert timeseries.shape == (100,)
    assert label == "W0"
    assert jnp.all(
        timeseries == 64
    )  # Each element in sol is 1, summed over 4*4*4*1 = 64


def test_get_timeseries_from_solution_with_command_incidence():
    # Test case 4: Command is 'incidence'
    sol = _get_sol()
    compartment_idx, wane_idx, strain_idx = _get_index_enums()
    timeseries, label = utils.get_timeseries_from_solution_with_command(
        sol, compartment_idx, wane_idx, strain_idx, "incidence"
    )
    assert timeseries.shape == (99,)
    assert label == "E : incidence"
    assert jnp.all(
        timeseries == 0
    )  # Since the input arrays are all ones, the diff should be zeros


def test_get_timeseries_from_solution_with_command_strain_prevalence():
    # Test case 5: Command is 'strain_prevalence'
    sol = _get_sol()
    compartment_idx, wane_idx, strain_idx = _get_index_enums()
    timeseries, labels = utils.get_timeseries_from_solution_with_command(
        sol, compartment_idx, wane_idx, strain_idx, "strain_prevalence"
    )
    assert len(timeseries) == 4
    assert len(labels) == 4
    assert jnp.all(
        jnp.array([jnp.array(j).shape == (100,) for j in timeseries])
    )
    assert set(labels) == set(strain_idx._member_names_)


def test_get_timeseries_from_solution_with_command_compartment_slice():
    # Test case 6: Command is a slice of a compartment
    sol = _get_sol()
    compartment_idx, wane_idx, strain_idx = _get_index_enums()
    timeseries, label = utils.get_timeseries_from_solution_with_command(
        sol, compartment_idx, wane_idx, strain_idx, "S[:, 0, 0, :]"
    )
    assert timeseries.shape == (100,)
    assert label == "S[:, 0, 0, :]"
    assert jnp.all(
        timeseries == 16
    )  # Each element in sol is 1, summed over 4*1*1*4 = 16
