"""A utils file full of different utility functions used within various components of Initialization, inference, running, and interpretation"""

import datetime
import glob
import os
import sys

# importing under a different name because mypy static type hinter
# strongly dislikes the IntEnum class.
from enum import EnumMeta as IntEnum
from typing import Any

import epiweeks
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro  # type: ignore
import numpyro.distributions as Dist  # type: ignore
import pandas as pd  # type: ignore
from jax import Array
from scipy.stats import gamma

pd.options.mode.chained_assignment = None


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# SAMPLING FUNCTIONS
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
def sample_if_distribution(parameters):
    """
    given a dictionary of keys and parameters, searches through all keys
    and samples the distribution associated with that key, if it exists.
    Otherwise keeps the value associated with that key.
    Converts lists with distributions inside to `jnp.ndarray`

    Lists containing distributions will have the parameter's index
    marked according to its position in the matrix.
    For some 2x2 matrix `x`, `x_1_1` refers to the sampled
    version of the last element of `x`

    Parameters
    ----------
    `parameters: dict{str: obj}`
    a dictionary mapping a parameter name to an object, either a value or a distribution.
    `numpyro.distribution` objects are sampled, and their sampled value replaces the distribution object
    within parameters. Capable of sampling lists with static values and distributions together.

    Returns
    ----------
    parameters dictionary with any `numpyro.distribution` objects replaced with jax.tracer samples
    of those distributions from `numpyro.sample`
    """
    for key, param in parameters.items():
        # if distribution, sample and replace
        if issubclass(type(param), Dist.Distribution):
            param = numpyro.sample(key, param)
        # if list, check for distributions within and replace them
        elif isinstance(param, (np.ndarray, list)):
            param = np.array(param)  # cast np.array so we get .shape
            flat_param = np.ravel(param)  # Flatten the parameter array
            # check for distributions inside of the flattened parameter list
            if any(
                [
                    issubclass(type(param_lst), Dist.Distribution)
                    for param_lst in flat_param
                ]
            ):
                dim_idxs = np.unravel_index(
                    np.arange(flat_param.size), param.shape
                )
                # if we find distributions, sample them, then reshape back to the original shape
                # all this code with dim_idxs and joining strings is to properly display the
                # row/col indexes in any number of dimensions, not just 1 and 2D matrix
                flat_param = jnp.array(
                    [
                        (
                            numpyro.sample(
                                key
                                + "_"
                                + "_".join(
                                    [str(dim_idx[i]) for dim_idx in dim_idxs]
                                ),
                                param_lst,
                            )
                            if issubclass(type(param_lst), Dist.Distribution)
                            else param_lst
                        )
                        for i, param_lst in enumerate(flat_param)
                    ]
                )
                param = jnp.reshape(flat_param, param.shape)
        # else static param, do nothing
        parameters[key] = param
    return parameters


def identify_distribution_indexes(
    parameters: dict[str, Any],
) -> dict[str, dict[str, str | tuple | None]]:
    """
    A inverse of the `sample_if_distribution()` which allows users to identify the locations
    of numpyro samples. Given a dictionary of parameters, identifies which parameters
    are numpyro distributions or are distributions within a list and returns a mapping
    between the sample names and its actual parameter name and index.

    Example
    --------------
    parameters = {"test": [0, numpyro.distributions.Normal(), 2], "example": numpyro.distributions.Normal()}
    identify_distribution_indexes(parameters) = {"test_1": {"sample_name": "test", "sample_idx": tuple(1)},
                                                 "example":  {"sample_name": "example", "sample_idx": None}}

    Parameters
    -------------
    a dictionary containing keys of different parameters names and values of any type

    Returns
    ------------
    `dict[str, dict[str, str | tuple[int] | None]]`

    a dictionary mapping the sample name to the parameter name within `parameters`.
    (if the sampled parameter is within a larger list, returns a tuple of indexes as well, otherwise None)
    key: str -> sampled parameter name as produced by `sample_if_distribution()`
    value: `dict[str, str | tuple | None]` -> "sample_name" = sample name within input `parameters`
                                      -> "sample_idx" = sample index if within list, else None
    """

    def get_index(indexes):
        return tuple(indexes)

    index_locations: dict[str, dict[str, str | tuple | None]] = {}
    for key, param in parameters.items():
        # if distribution, it does not have an index, so None
        if issubclass(type(param), Dist.Distribution):
            index_locations[key] = {"sample_name": key, "sample_idx": None}
        # if list, check for distributions within and mark their indexes
        elif isinstance(param, (np.ndarray, list)):
            param = np.array(param)  # cast np.array so we get .shape
            flat_param = np.ravel(param)  # Flatten the parameter array
            # check for distributions inside of the flattened parameter list
            if any(
                [
                    issubclass(type(param_lst), Dist.Distribution)
                    for param_lst in flat_param
                ]
            ):
                dim_idxs = np.unravel_index(
                    np.arange(flat_param.size), param.shape
                )
                for i, param_lst in enumerate(flat_param):
                    if issubclass(type(param_lst), Dist.Distribution):
                        param_idxs = [dim_idx[i] for dim_idx in dim_idxs]
                        index_locations[
                            str(
                                key
                                + "_"
                                + "_".join(
                                    [str(dim_idx[i]) for dim_idx in dim_idxs]
                                )
                            )
                        ] = {
                            "sample_name": key,
                            "sample_idx": get_index(param_idxs),
                        }
    return index_locations


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# SPLINE FUNCTIONS
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


# Vaccination modeling, using cubic splines to model vax uptake in the population stratified by age and current vax shot.
def base_equation(t, coefficients):
    """
    the base of a spline equation, without knots, follows a simple cubic formula
    a + bt + ct^2 + dt^3. This is a vectorized version of this equation which takes in
    a matrix of `a` values, as well as a marix of `b`, `c`, and `d` coefficients.
    PARAMETERS
    ----------
    t: jax.tracer array
        a jax tracer containing within it the time in days since model simulation start
    intercepts: jnp.array()
        intercepts of each cubic spline base equation for all combinations of age bin and vax history
        intercepts.shape=(NUM_AGE_GROUPS, MAX_VACCINATION_COUNT + 1)
    coefficients: jnp.array()
        coefficients of each cubic spline base equation for all combinations of age bin and vax history
        coefficients.shape=(NUM_AGE_GROUPS, MAX_VACCINATION_COUNT + 1, 3)
    """
    return jnp.sum(
        coefficients
        * jnp.array([1, t, t**2, t**3])[jnp.newaxis, jnp.newaxis, :],
        axis=-1,
    )


def conditional_knots(t, knots, coefficients):
    indicators = jnp.where(t > knots, t - knots, 0)
    # multiply coefficients by 3 since we taking derivative of cubic spline.
    return jnp.sum(indicators**3 * coefficients, axis=-1)


# days of separation between each knot


def evaluate_cubic_spline(
    t,
    knot_locations: jnp.ndarray,
    base_equations: jnp.ndarray,
    knot_coefficients: jnp.ndarray,
) -> float:
    """
    Returns the value of a cubic spline with knots and coefficients evaluated on day `t` for each age_bin x vax history combination.
    Cubic spline equation:

    f(t) = a + bt + ct^2 + dt^3 + sum_{i}^{len(knots)}(knot_coefficients_{i} * (t-knot_locations_{i})^3 * I(t > knot_locations_{i}))

    Where coef/knots[i] is the i'th index of each array. and the I() function is an indicator variable 1 or 0.

    Parameters
    ----------
    t: jax.tracer array
        a jax tracer containing within it the time in days since model simulation start
    knot_locations: jnp.ndarray
        knot locations of each cubic spline for all combinations of age bin and vax history
        knots.shape=(NUM_AGE_GROUPS, MAX_VACCINATION_COUNT + 1, # knots in each spline)
    base_equations" jnp.ndarray
        the base equation coefficients (a + bt + ct^2 + dt^3) of each cubic spline for all combinations of age bin and vax history
        knots.shape=(NUM_AGE_GROUPS, MAX_VACCINATION_COUNT + 1, 4)
    knot_coefficients: jnp.ndarray
        knot coefficients of each cubic spline for all combinations of age bin and vax history.
        including first 4 coefficients for the base equation.
        coefficients.shape=(NUM_AGE_GROUPS, MAX_VACCINATION_COUNT + 1, # knots in each spline + 4)

    Returns
    ----------
    jnp.array() containing the proportion of individuals in each age x vax combination that will be vaccinated during this time step.
    """
    base = base_equation(t, base_equations)
    knots = conditional_knots(t, knot_locations, knot_coefficients)
    return base + knots


def season_1peak(t, seasonality_second_wave, seasonality_shift):
    """
    a utils function used to calculate seasonality,
    this one is for the winter wave occuring at t=0 if `seasonality_shift=0`
    and `seasonality_second_wave=0`
    """
    return (1 - seasonality_second_wave) * jnp.cos(
        2 * jnp.pi * (t - seasonality_shift) / 365
    )


def season_2peak(t, seasonality_second_wave, seasonality_shift):
    """
    a utils function used to calculate seasonality,
    this one is for the summer wave occuring at t=182.5 if `seasonality_shift=0`
    and `seasonality_second_wave=1`
    """
    return seasonality_second_wave * jnp.cos(
        4 * jnp.pi * (t - seasonality_shift) / 365
    )


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# INDEXING FUNCTIONS
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


def sim_day_to_date(sim_day: int, init_date: datetime.date):
    """
    given the current model's simulation day as an integer, and the date of the model initialization.
    returns a date object representing the current simulation day.

    Parameters
    ----------
    sim_day: int
        current model simulation day where sim_day==0==init_date
    init_date: datetime.date
        the initialization date of the simulation, usually found in the config.INIT_DATE parameter

    Returns
    -----------
    datetime.date object representing the current sim_day of the simulation.
    """
    return init_date + datetime.timedelta(days=sim_day)


def sim_day_to_epiweek(sim_day: int, init_date: datetime.date):
    """
    given the current model's simulation day as an integer, and the date of the model initialization.
    returns an integer cdc epiweek that sim_day falls in.

    Parameters
    ----------
    sim_day: int
        current model simulation day where sim_day==0==init_date
    init_date: datetime.date
        the initialization date of the simulation, usually found in the config.INIT_DATE parameter

    Returns
    -----------
    epiweek.Week object representing the cdc epiweek of the simulation on day sim_day.
    """
    date = sim_day_to_date(sim_day, init_date)
    epi_week = epiweeks.Week.fromdate(date)
    return epi_week


def date_to_sim_day(date: datetime.date, init_date: datetime.date):
    """
    given a date object, converts back to simulation days using init_date as reference for t=0

    Parameters
    ----------
    sim_day: datetime.date
        date to be converted to a simulation day
    init_date: datetime.date
        the initialization date of the simulation, usually found in the config.INIT_DATE parameter

    Returns
    -----------
    int simulation day representing how many days from `init_date` have passed.
    """
    return (date - init_date).days


def date_to_epi_week(date: datetime.date):
    """
    given a date object, converts to cdc epi week using init_date as reference for t=0

    Parameters
    ----------
    sim_day: datetime.date
        date to be converted to a simulation day
    Returns
    -----------
    epiweeks.Week obj representing the epi_week that `date` falls in
    """
    epi_week = epiweeks.Week.fromdate(date)
    return epi_week


def new_immune_state(current_state: int, exposed_strain: int) -> int:
    """a method using BITWISE OR to determine a new immune state position given
    current state and the exposing strain

    Parameters
    ----------
    current_state: int
        int representing the current state of the individual or group being exposed to a strain
    exposed_strain: int
        int representing the strain exposed to the individuals in state `current_state`.
        expects that `0 <= exposed_strain <= num_strains - 1`
    num_strains: int
        number of strains in the model

    Example
    ----------
    num_strains = 2, possible states are:
    00(no exposure), 1(exposed to strain 0 only), 2(exposed to strain 1 only), 3(exposed to both)

    new_immune_state(current_state, exposed_strain): new_state (explanation)
    new_immune_state(0, 0): 1 (no previous exposure, now exposed to strain 0)
    new_immune_state(0, 1): 2 (no previous exposure, now exposed to strain 1)
    new_immune_state(1, 0): 1 (exposed to strain 0 already, no change in state)
    new_immune_state(2, 1): 2 (exposed to strain 1 already, no change in state)
    new_immune_state(1, 1): 3 (exposed to strain 0 prev, now exposed to both)
    new_immune_state(2, 0): 3 (exposed to strain 1 prev, now exposed to both)
    new_immune_state(3, 0): 3 (exposed to both already, no change in state)
    new_immune_state(3, 1): 3 (exposed to both already, no change in state)
    """
    if isinstance(exposed_strain, (int, float)) and isinstance(
        current_state, (int, float)
    ):
        # if using ints and floats, stay in int land and BITWISE OR them
        current_state_binary = format(current_state, "b")
        exposed_strain_binary = format(2**exposed_strain, "b")
        new_state = format(
            int(current_state_binary, 2) | int(exposed_strain_binary, 2), "b"
        )
        return int(new_state, 2)
    else:  # being used with jax tracers
        # if we are passing jax tracers, convert to bit arrays first
        current_state_binary = jnp.unpackbits(
            jnp.array([current_state]).astype("uint8")
        )
        exposing_strain_binary = jnp.unpackbits(
            jnp.array([2**exposed_strain]).astype("uint8")
        )
        return jnp.packbits(
            jnp.bitwise_or(current_state_binary, exposing_strain_binary)
        )


def all_immune_states_with(strain: int, num_strains: int):
    """
    a function returning all of the immune states which contain an exposure to `strain`

    Parameters
    ----------
    strain: int
        int representing the exposed to strain, expects that `0 <= strain <= num_strains - 1`
    num_strains: int
        number of strains in the model

    Returns
    ----------
    list[int] representing all states that include previous exposure to `strain`

    Example
    ----------
    in a simple model where num_strains = 2 the following is returned.
    Reminder: state = 0 (no exposure),
    state = 1/2 (exposure to strain 0/1 respectively), state=3 (exposed to both)

    all_immune_states_with(0, 2) -> [1, 3]

    all_immune_states_with(1, 2) -> [2, 3]
    """
    # represent all possible states as binary
    binary_array = [bin(val) for val in range(2**num_strains)]
    # represent exposing strain as an indiciator bit string. ex: exposing_strain = 1 -> binary = 10
    strain_binary_lst = ["0"] * num_strains
    strain_binary_lst[-(strain + 1)] = "1"
    strain_binary = "".join(strain_binary_lst)
    # a state contains the strain being filtered if the bitwise AND produces a non-zero value
    filtered_states = [
        int(binary, 2)
        for binary in binary_array
        if (int(binary, 2) & int(strain_binary, 2)) > 0
    ]
    return filtered_states


def all_immune_states_without(strain: int, num_strains: int):
    """
    function returning all of the immune states which DO NOT contain an exposure to `strain`

    Parameters
    ----------
    strain: int
        int representing the NOT exposed to strain, expects that `0 <= strain <= num_strains - 1`
    num_strains: int
        number of strains in the model

    Returns
    ----------
    list[int] representing all states that DO NOT include previous exposure to `strain`

    Example
    ----------
    in a simple model where num_strains = 2 the following is returned.
    Reminder: state = 0 (no exposure),
    state = 1/2 (exposure to strain 0/1 respectively), state=3 (exposed to both)

    all_immune_states_without(strain = 0, num_strains = 2) -> [0, 2]

    all_immune_states_without(strain = 1, num_strains = 2) -> [0, 1]
    """
    all_states = list(range(2**num_strains))
    states_with_strain = all_immune_states_with(strain, num_strains)
    # return set difference of all states and states including strain
    return list(set(all_states) - set(states_with_strain))


def get_strains_exposed_to(state: int, num_strains: int):
    """
    Returns a list of integers representing the strains a given individual was exposed to end up in state `state`.
    Says nothing of the order at which an individual was exposed to those strains, list returned sorted increasing.

    Parameters
    -----------
    state: int
        the state a given individual is in, as dicated by a single or series of exposures to strains.
        state dynamics determined by `new_immune_state()`
    num_strains: int
        the total number of strains in the model, used to determin total size of state space.

    Returns
    -----------
        list[int] representing the strains the individual in `state` was exposed to, sorted increasing.
    """
    state_binary = format(state, "b")
    # prepend 0s if needed.
    if len(state_binary) < num_strains:
        state_binary = "0" * (num_strains - len(state_binary)) + state_binary
    # inverse order since index 0 will be at the end of the str after prepending 0s
    state_binary_lst = list(state_binary)[::-1]
    # if val == 1 in the binary, that means that state was exposed to that strain.
    # strain is marked by the index of each 1 in the list.
    strains_exposed_by = [
        i for i in range(len(state_binary_lst)) if state_binary_lst[i] == "1"
    ]
    return strains_exposed_by


def combined_strains_mapping(
    from_strain: int, to_strain: int, num_strains: int
):
    """
    given a strain `from_strain` and `to_strain` returns a mapping of all immune states before and after strains are combined.

    Example
    -----------
    in a basic 2 strain model you have the following immune states:
    0-> no exposure, 1 -> strain 0 exposure, 2-> strain 1 exposure, 3-> exposure to both

    calling `combine_strains(1, 0, 2)` will combine strains 0 and 1 returning
    `{0:0, 1:1, 2:1, 3:1}`, because there is no functional difference strain 0 and 1 the immune state space becomes binary.

    Parameters
    ----------
    from_strain: int
        the strain index representing the strain being collapsed, whos references will be rerouted.
    to_strain: int
        the strain index representing the strain being joined with to_strain, typically the ancestral or 0 index.

    Returns
    -----------
    dict[int:int] mapping from immune state -> immune state before and after `from_strain` is combined with `to_strain` for all states.

    dict[int:int] mapping from strain idx -> strain idx before and after`from_strain` is combined with `to_strain` for all strains.
    """
    # we do nothing if from_strain is equal to to_strain, we arent collapsing anything there.
    if from_strain == to_strain:
        return {x: x for x in range(2**num_strains)}, {
            x: x for x in range(num_strains)
        }

    # create a helper function so we can pass old strains and have it auto-convert.
    def translate_strain(strain_in):
        if strain_in == from_strain:
            return to_strain
        elif strain_in > from_strain:
            return strain_in - 1
        return strain_in

    # maps old immune state to new immune state
    immune_state_converter = {0: 0}  # 0 -> 0 always
    strain_converter = {
        strain: translate_strain(strain) for strain in range(num_strains)
    }
    # go through each state, break apart into strain hist, use `translate_strain`, recreate a new state
    for immune_state in range(2**num_strains):
        old_strains_in_state = get_strains_exposed_to(
            immune_state, num_strains
        )
        collapsed_strains_in_state = [
            translate_strain(strain) for strain in old_strains_in_state
        ]
        new_state = 0  # init at no exposures
        # build new state with the collapsed strain indexes, one at a time
        for new_strain in collapsed_strains_in_state:
            # expose new_state to the redefined strain definitions
            new_state = new_immune_state(new_state, new_strain)
        # all individuals in `immune_state` before strain definition collapse
        # are now in `new_state`
        immune_state_converter[immune_state] = new_state
    return immune_state_converter, strain_converter


def combine_strains(
    compartment: np.ndarray,
    state_mapping: dict[int, int],
    strain_mapping: dict[int, int],
    num_strains: int,
    state_dim=1,
    strain_dim=3,
    strain_axis=False,
):
    """
    takes an individual compartment and combines the states and strains within it according to `state_mapping` and `strain_mapping`.
    If compartment has a strain axis, strain_axis=True.

    Parameters
    ----------
    compartment: np.ndarray
        the compartment being changed, must be four dimensional with immune state in the 2nd dimension and strain (if applicable) in the last dimension.
    state_mapping: dict[int:int]
        a mapping of pre-combine state to post-combine state, as generated by combined_strains_mapping(), must cover all states found in `compartment`.
        can be many to one relationship of keys to state values.
    strain_mapping: dict[int:int]
        a mapping of pre-combine strain to post-combine strain, as generated by combined_strains_mapping(), must cover all strains found in `compartment`.
        can be many to one relationship of keys to state values.
    num_strains: int
        number of strains in the model
    state_dim: int
        if the dimension of the immune_state column is non-standard, specify which dimension immune state is found in
    strain_dim: int
        if the dimension of the strain column is non-standard, specify which dimension strain is found in
    strain_axis: bool
        whether or not `compartment` includes a strain axis in the last dimension that must also be combined.

    Returns
    ----------
    np.ndarray:
        a modified copy of `compartment` with all immune states and strains combined according to state_mapping and strain_mapping
    """
    # begin with a copy of the compartment in all zeros
    strain_combined_compartment = np.zeros(compartment.shape)
    # next update the immune states according to the state_mapping dict
    for immune_state in range(2**num_strains):
        # after strain combining immune_state moves to `new_state`
        new_state = state_mapping[immune_state]
        # += because multiple `immune_states` can flow into one `new_state`
        # use swapaxis to grab an arbitrary dimension of the array, ignoring ordering bugs
        strain_combined_compartment.swapaxes(0, state_dim)[
            new_state
        ] += compartment.swapaxes(0, state_dim)[immune_state]
    # next, if we must also remap an infected_by strain axis, do that
    if strain_axis:
        # anything that does not have a strain flowing into it, ends up being zeroed out
        # this is because strain_combined_compartment has data prepopulated from the state reshuffle above
        zero_out = set(range(num_strains))
        # go through each strain and reshuffle it according to strain_mapping dict
        for strain in range(num_strains):
            to_strain = strain_mapping[strain]
            zero_out = zero_out - set([to_strain])
            # if strain=to_strain avoid double addition since data is already there.
            if strain != to_strain:
                # use swapaxis to grab an arbitrary dimension of the array, ignoring ordering bugs
                strain_combined_compartment.swapaxes(0, strain_dim)[
                    to_strain
                ] += strain_combined_compartment.swapaxes(0, strain_dim)[
                    strain
                ]
                # once we moved individuals with this strain into the correct compartment, we zero it out.
                # otherwise we have doubled the population.
                strain_combined_compartment.swapaxes(0, strain_dim)[strain] = 0
        # finally zero out the spaces left behind after the shuffle
        # use swapaxis to grab an arbitrary dimension of the array, ignoring ordering bugs
        strain_combined_compartment.swapaxes(0, strain_dim)[list(zero_out)] = 0
    return strain_combined_compartment


def combine_epochs(
    epoch_solutions, from_strains, to_strains, strain_idxs, num_tracked_strains
):
    """
    given N epochs, combines their solutions by translating all immune states and infections of past epochs into the most recent epochs defintions.
    Solutions are expected to be 5 dimensions, with the first dimension being timesteps, and the remaining 4 following the standard compartment structure.
    immune state in the 3rd dimension (of 5) and strain in the last dimension (if applicable).

    `epoch_solutions`, `from_strains`, and `to_strains` must be given in order from earliest to most recent epoch.
    Does not assume anything about the dates or times these events occured on other than them being sequential.

    Parameters
    ----------
    epoch_solutions: list[tuple(np.ndarray)]
        a list of each epoch's solution.ys object as given by Diffeqsolve().ys or BasicMechanisticModel.run().ys
        in order from earliest to most recent epoch.

    from_strains: list[int/None]
        a parallel list of strain indexes indiciating the strain combinations that occured at the end of each epoch.
        len(from_strain) = N-1 for N epochs, since last epoch does not combine with anything

    to_strains: list[int/None]
        a parallel list of strain indexes indiciating the strain combinations that occured at the end of each epoch.
        len(to_strain) = N-1 for N epochs, since last epoch does not combine with anything

    strain_idxs: list[IntEnum]
        a list of IntEnums to identify the strain name to index for each epoch.
        len(strain_idxs) = N for N epochs

    num_strains_consistent: int
        the number of strains consistent across all epochs, their definitions may change but there are always `num_tracked_strains` tracked in each epoch.

    Returns
    -----------
    tuple(np.ndarray): a single state object that combines the timelines of all N epochs with states and strain definitions matching that of the most recent epoch.
    """

    transition_tables = []
    # create transition tables for each epoch to the next
    for idx, (from_strain, to_strain) in enumerate(
        zip(from_strains, to_strains)
    ):
        from_strain = strain_idxs[idx][from_strain]
        to_strain = strain_idxs[idx][to_strain]
        if from_strains is None or to_strains is None:
            transition_tables.append(transition_tables)
        transition_tables.append(
            combined_strains_mapping(
                from_strain, to_strain, num_tracked_strains
            )
        )

    # yes the following code is O(N**2) for N epochs, N is supposed to be small
    # modify each transition table so it translates states directly into the last state
    # skip the last table since it does not need to scan multiple
    for idx, (state_mapping, strain_mapping) in enumerate(
        transition_tables[:-1]
    ):
        # use a loop to update the value according to the future transition tables
        # example state 7 -> 3 -> 1 through two strain combinations
        # example strain 2-> 1 -> 0 through two strain combinations
        for future_state_mapping, future_strain_mapping in transition_tables[
            idx + 1 :
        ]:
            for state in state_mapping.keys():
                state_mapping[state] = future_state_mapping[
                    state_mapping[state]
                ]
            for strain in strain_mapping.keys():
                strain_mapping[strain] = future_strain_mapping[
                    state_mapping[strain]
                ]
    # we now have transition tables mapping every epoch solution to the states of the last one!
    # next we must create the final solutions tuple, combining each epoch and appending it to the last
    num_compartments_per_solution = len(epoch_solutions[0])
    total_timeline_state = [False] * num_compartments_per_solution
    for (state_mapping, strain_mapping), solution in zip(
        transition_tables, epoch_solutions[:-1]
    ):
        for idx, compartment in enumerate(solution):
            strain_axis = idx != 0
            # our compartments have an extra dimension prepended, so we specify state_dim and strain_dim.
            epoch_solution = combine_strains(
                compartment,
                state_mapping,
                strain_mapping,
                num_tracked_strains,
                state_dim=2,
                strain_dim=4,
                strain_axis=strain_axis,
            )
            current_solution = total_timeline_state[idx]
            if current_solution:  # if we have one to merge with
                # skip first row since current_solution[-1] == epoch_solution[0]
                # since it is used as initialization
                current_solution = np.concatenate(
                    [current_solution, epoch_solution[1:]], axis=(0)
                )
            else:
                total_timeline_state[idx] = epoch_solution
    # last but not least, append the last epoch onto each of the compartments
    for compartment_idx in range(num_compartments_per_solution):
        epoch_solution = epoch_solutions[-1][compartment_idx]
        total_timeline_state[compartment_idx] = np.concatenate(
            [total_timeline_state[compartment_idx], epoch_solution[1:]],
            axis=(0),
        )

    return tuple(total_timeline_state)


def find_age_bin(age: int, age_limits: list[int]) -> int:
    """
    Given an age, return the age bin it belongs to in the age limits array

    Parameters
    ----------
    age: int
        age of the individual to be binned
    age_limits: list(int)
        age limit for each age bin in the model, begining with minimum age
        values are exclusive in upper bound. so [0,18) means 0-17, 18+

    Returns
    ----------
    The index of the bin, assuming 0 is the youngest age bin and len(age_limits)-1 is the oldest age bin
    """
    current_bin = -1
    for age_limit in age_limits:
        if age - age_limit < 0:
            return current_bin
        else:
            current_bin += 1
    return current_bin


def find_vax_bin(vax_shots: int, max_doses: int) -> int:
    """
    Given a number of vaccinations, returns the bin it belongs to given the maximum doses ceiling

    Parameters
    ----------
    vax_shots: int
        the number of vaccinations given to the individual
    max_doses: int
        the number of doses maximum before all subsequent doses are no longer counted

    Returns
    ----------
    The index of the vax bin, min(vax_shots, max_doses)
    """
    return min(vax_shots, max_doses)


def convert_hist(strains: str, STRAIN_IDX: IntEnum) -> int:
    """
    a function that transforms a comma separated list of strains and transform them into an immune history state.
    Any unrecognized strain strings inside of `strains` do not contiribute to the returned state.

    Example
    ----------
    strains: "alpha, delta, omicron"
    STRAIN_IDX: delta=0, omicron=1
    num_strains: 2

    method will ignore alpha infection as it is not in STRAIN_IDX, returning state=3, indicating infection with both delta and omicron.

    Parameters
    ----------
    strains: str
        a comma separated string of each exposed strain, order does not matter, capitalization does not matter.
    STRAIN_IDX: intEnum
        an enum containing the name of each strain and its associated strain index, as initialized by ConfigBase.
    num_strains:
        the number of _tracked_ strains in the model.

    """
    state = 0
    for strain in filter(None, strains.split(",")):
        strain_idx = convert_strain(strain, STRAIN_IDX)
        state = new_immune_state(state, strain_idx)
    return state


def convert_strain(strain: str, STRAIN_IDX: IntEnum) -> int:
    """
    given a text description of a string, return the correct strain index as specified by the STRAIN_IDX enum.
    If strain is not found in STRAIN_IDX, return 0 (the oldest strain included in the model)

    Parameters
    -----------
    strain: str
        a string representing the infecting strain, capitalization does not matter.
    STRAIN_IDX: intEnum
        an enum containing the name of each strain and its associated strain index, as initialized by ConfigBase.

    Returns
    ----------
    STRAIN_IDX[strain] if exists, else 0
    """
    if strain.lower() in STRAIN_IDX._member_map_:
        return STRAIN_IDX[strain.lower()]
    else:
        return 0  # return oldest strain if not included


def find_waning_compartment(TSLIE: int, waning_times: list[int]) -> int:
    """
    Given a TSLIE (time since last immunogenetic event) in days, returns the waning compartment index of the event.

    Parameters
    ----------
    TSLIE: int
        the number of days since the initialization of the model that the immunogenetic event occured (this could be vaccination or infection).
    waning_times: list(int)
        the number of days an individual stays in each waning compartment, ending in zero as the last compartment does not wane.

    Returns
    ----------
    index of the waning compartment that an event belongs, to if that event happened `TSLIE` days in the past.
    """
    # possible with cumulative sum, but this solution still O(N) and more interpretable
    current_bin = 0
    for wane_time in waning_times:
        if TSLIE - wane_time < 0:
            return current_bin
        else:
            TSLIE -= wane_time
            current_bin += 1
    # last compartment waning_time = 0, shifts us 1 extra bin, shift back in this edge case.
    return current_bin - 1


def strain_interaction_to_cross_immunity(
    num_strains: int, strain_interactions: np.ndarray
) -> Array:
    """
    a function which takes a strain_interactions matrix, which is of shape (num_strains, num_strains)
    and returns a cross immunity matrix of shape (num_strains, 2**num_strains) representing the immunity
    of all 2**num_strains immune histories against some challenging strain.

    Parameters
    ----------
    num_strains: int
        the number of strains for which the crossimmunity matrix is being generated.
    strain_interactions: np.array
        a matrix of shape (num_strains, num_strains) representing the relative immunity of someone recovered from
        one strain to a different challenging strain. 1's in the diagnal representing 0 reinfection (before waning).

    Returns
    ----------
    crossimmunity_matrix: jnp.array
        a matrix of shape (num_strains, 2**num_strains) representing the relative immunity of someone with a specific
        immune history to a challenging strain.
    """
    infection_history = range(2**num_strains)
    crossimmunity_matrix = jnp.zeros((num_strains, len(infection_history)))
    for challenging_strain in range(num_strains):
        # if immune history already contains exposure to challenging_strain, this is a reinfection.
        crossimmunity_matrix = crossimmunity_matrix.at[
            challenging_strain,
            all_immune_states_with(challenging_strain, num_strains),
        ].set(strain_interactions[challenging_strain, challenging_strain])
        # for individuals without previous exposure to this strain, use protection from most recent infection.
        states_without_strain = all_immune_states_without(
            challenging_strain, num_strains
        )
        for state_without_strain in states_without_strain:
            # if state = 0, they have no most recent infection, thus 0 immunity
            if state_without_strain == 0:
                crossimmunity_matrix = crossimmunity_matrix.at[
                    challenging_strain, state_without_strain
                ].set(0)
            else:  # find last most recent infection
                # turn state into binary, find the 1 correlating to the most recent strain
                state_binary = str(bin(state_without_strain))[2:]  # 0b remove
                # prepend 0s to make string correct length eg 10 -> 010 in 3 strain model
                state_binary = (
                    state_binary
                    if len(state_binary) == num_strains
                    else ("0" * (num_strains - len(state_binary)))
                    + state_binary
                )
                # convert the state to a list of strain exposures
                state_list = [int(d) for d in state_binary][::-1]
                strains = np.where(np.array(state_list) == 1)[0]
                # find the most recent strain that has been exposed to
                # this is often the same thing as the most recent exposed strain
                # people can be reinfected by older strains after newer ones
                most_recent_immune_strain = strains[np.argmax(strains)]
                crossimmunity_matrix = crossimmunity_matrix.at[
                    challenging_strain, state_without_strain
                ].set(
                    strain_interactions[
                        challenging_strain, most_recent_immune_strain
                    ]
                )
    return crossimmunity_matrix


def drop_sample_chains(samples: dict, dropped_chain_vals: list):
    """
    a function, given a dictionary which is the result of a call to `mcmc.get_samples()`
    drops specified chains from the posterior samples. This is usually done when a single or multiple
    chains do not converge with the other chains. This ensures that this divergent chain does not
    impact posterior distributions meant to summarize the posterior samples.

    Parameters
    -----------
    `samples`: dict{str: list}
        a dictionary where parameter names are keys and samples are a list.
        In the case of M chains and N samples per chain, the list will be of shape MxN
        with one row per chain, each containing N samples.

    `dropped_chain_vals`: list
        a list of indexes (rows in the MxN grouped samples list) to be dropped,
        if the list is empty no chains are dropped.

    Returns
    ----------
    dict{str: list} a copy of the samples dictionary with chains in `dropped_chain_vals` dropped
    """
    # Create a new dictionary to store the filtered samples
    filtered_dict = {}
    # Iterate over the keys (parameter names) in the original dictionary
    for param_name in samples.keys():
        # Get the samples for the current parameter
        param_samples = samples[param_name]
        # Remove the specified chains from the samples
        filtered_samples = np.array(
            [
                samples
                for i, samples in enumerate(param_samples)
                if i not in dropped_chain_vals
            ]
        )
        # Add the filtered samples to the new dictionary
        filtered_dict[param_name] = filtered_samples

    return filtered_dict


def flatten_list_parameters(
    samples: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """
    given a dictionary of parameter names and samples, identifies any parameters that are
    placed under a single name, but actually multiple independent draws from the same distribution.
    These parameters are often the result of a call to `numpyro.plate(P)` for some number of draws `P`
    After identifying plated samples, this function will separate the `P` draws into their own
    keys in the samples dictionary.

    Parameters
    ----------
    `samples`: dict{str: np.ndarray}
        a dictionary where parameter names are keys and samples are a list.
        In the case of M chains and N samples per chain, the list will be of shape MxN normally
        with one row per chain, each containing N samples.
        In the case that the parameter is drawn P independent times, the list will be of shape
        MxNxP.

    Returns
    ----------
    dict{str: np.ndarray}  a dictionary in which parameters with lists of shape MxNxP are split into
    P separate parameters, each with lists of shape MxN for M chains and N samples.
    This function scaled with any number of dimensions > 2. So for PxQ independent draws
    will be separated into PxQ parameters each with lists of shape MxN.

    NOTE
    -----------
    If you only have parameters of dimension 2, nothing will be changed and a copy of your dict will be returned
    """
    return_dict = {}
    for key, value in samples.items():
        if isinstance(value, np.ndarray) and value.ndim > 2:
            num_dims = value.ndim - 2
            indices = (
                np.indices(value.shape[-num_dims:]).reshape(num_dims, -1).T
            )

            for idx in indices:
                new_key = f"{key}"
                for i in range(len(idx)):
                    new_key += f"_{idx[i]}"

                new_value = value[
                    tuple([slice(None)] * (value.ndim - num_dims) + list(idx))
                ]
                return_dict[new_key] = new_value
        else:
            return_dict[key] = value
    return return_dict


def drop_keys_with_substring(dct: dict[str, Any], drop_s: str):
    """A simple helper function designed to drop keys from a dictionary if they contain some substring

    Parameters
    ----------
    dct : dict[str, Any]
        a dictionary with string keys
    drop_s : str
        keys containing `drop_s` as a substring will be dropped

    Returns
    -------
    dict[str, any]
        dct with keys containing drop_s removed, otherwise untouched.
    """
    keys_to_drop = [key for key in dct.keys() if drop_s in key]
    for key in keys_to_drop:
        del dct[key]
    return dct


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# DEATH CALCULATION CODE
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


def convolve_hosp_to_death(hosp, hfr, shape, scale, padding="nan"):
    """
    Model deaths based on hospitalizations. The function calculates expected deaths based
    on input weekly age-specific `hospitalization` and hospitalization fatality risk
    (`hfr`), then delay the deaths (relative to hospitalization) based on a gamma
    distribution of parameters `shape` and `scale`. The gamma specification is _daily_,
    which then gets discretized into 5 weeks for convolution.
    Parameters
    ----------
    `hosp` : numpy.array
        age-specific weekly hospitalization with shape of (num_weeks, NUM_AGE_GROUPS)
    `hfr`: numpy.array
        age-specific hospitalization fatality risk with shape of (NUM_AGE_GROUPS)
    shape: float
        shape parameter of the gamma delay distribution, is > 0
    scale: float
        scale parameter of the gamma delay distribution, is > 0 and 1/rate
    padding: str "nan", "nearest" or "no"
        boolean flag determining if the output array is of same length as `hosp` with
        first 4 weeks padded with nan or not. Note: the "valid" modelled deaths would always
        be 4 weeks less than input hospitalization.
    Returns
    ----------
    numpy.array:
        list of `num_day` vaccination rates arrays, each by the shape of (NUM_AGE_GROUPS,
        MAX_VAX_COUNT + 1)
    """
    expected_deaths = hosp * hfr[None, :]
    disc_gamma = gamma.cdf(np.arange(0, 36, 7), shape, scale=scale)
    disc_gamma = np.diff(disc_gamma)
    daily_deaths = np.array(
        [np.convolve(d, disc_gamma, "valid") for d in expected_deaths.T]
    ).T
    if padding == "nan":
        daily_deaths = np.append(
            np.array([[np.nan] * 4] * (len(disc_gamma) - 1)),
            daily_deaths,
            axis=0,
        )
    elif padding == "nearest":
        daily_deaths = np.append(
            np.repeat([daily_deaths[0]], len(disc_gamma) - 1, axis=0),
            daily_deaths,
            axis=0,
        )

    return daily_deaths


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# DEMOGRAPHICS CODE
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


def generate_yearly_age_bins_from_limits(age_limits: list) -> list[list[int]]:
    """
    given age limits, generates age bins with each year contained in that bin up to 85 years old exclusive

    Example
    ----------
    age_limits = [0, 5, 10, 15 ... 80]
    returns [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]... [80, 81, 82, 83, 84]]

    Parameters
    ----------
    age_limits: list(int):
        beginning with minimum age inclusive, boundary of each age bin exclusive. Not including last age bin.
        do not include implicit 85 in age_limits, this function appends that bin automatically.
    """
    age_groups = []
    for age_idx in range(1, len(age_limits)):
        age_groups.append(
            list(range(age_limits[age_idx - 1], age_limits[age_idx]))
        )
    age_groups.append(list(range(age_limits[-1], 85)))
    return age_groups


def load_age_demographics(
    path: str,
    regions: list[str],
    age_limits: list[int],
) -> dict[str, np.ndarray]:
    """Returns normalized proportions of each agebin as defined by age_limits for the regions given.
    Does this by searching for age demographics data in path.

    Parameters
    ----------
    path: str
        path to the demographic-data folder, either relative or absolute.
    regions: list(str)
        list of FIPS regions to create normalized proportions for
    age_limits: list(int)
        age limits for each age bin in the model, begining with minimum age
        values are exclusive in upper bound. so [0, 18, 50] means 0-17, 18-49, 50+
        max age is enforced at 84 inclusive. All persons older than 84 in population numbers are counted as 84 years old

    Returns
    ----------
    demographic_data : dict
        a dictionary maping FIPS code region supplied in `regions` to an array of length `len(age_limits)` representing
        the __relative__ population proportion of each bin, summing to 1.
    """
    assert os.path.exists(
        path
    ), "The path to population-rescaled age distributions does not exist as it should"

    demographic_data = {}
    # Create contact matrices
    for r in regions:
        try:
            # e.g., if territory is "North Carolina", pass it as "North_Carolina"
            if len(r.split()) > 1:
                region = "_".join(r.split())
            else:
                region = r
            if region != "United_States":
                region_data_file = (
                    "United_States_subnational_"
                    + region
                    + "_age_distribution_85.csv"
                )
            else:
                region_data_file = (
                    "United_States_country_level_age_distribution_85.csv"
                )

            age_distributions = np.loadtxt(
                path + region_data_file,
                delimiter=",",
                dtype=np.float64,
                skiprows=0,
            )
            binned_ages = np.array([])
            age_bin_pop = 0
            current_age = age_limits[0]
            if 84 not in age_limits:
                age_limits = age_limits + [84]
            age_limits = age_limits[1:]
            while current_age < 85:
                # get the population for the current age
                age_bin_pop += age_distributions[current_age][1]
                # add total population of that bin to the array, reset
                if current_age in age_limits:
                    binned_ages = np.append(binned_ages, age_bin_pop)
                    age_bin_pop = 0
                current_age += 1  # go to next year.
            # normalize array to proportions after all bins constructed.
            binned_ages = binned_ages / sum(binned_ages)
            demographic_data[r] = binned_ages
        except Exception as e:
            print(
                f"Something went wrong with {region} and produced the following error:\n\t{e}"
            )
    return demographic_data


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Plotting CODE
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


def plot_sample_chains(samples):
    """
    a function that given a dictionary of parameter names and MxN samples for each parameter
    plots the trace plot of each of the M chains through the N samples in that chain.

    Parameters
    ----------
    `samples`: dict{str: list}
        a dictionary where parameter names are keys and samples are a list.
        In the case of M chains and N samples per chain, the list will be of shape MxN
        with one row per chain, each containing N samples.

    Returns
    ----------
    plots each parameter along with each chain of that parameter,
    also returns `plt.fig` and `plt.axs` objects for modification.
    """
    # ensure samples are all NxM before plotting
    if any([samples[key].ndim == 3 for key in samples.keys()]):
        samples = flatten_list_parameters(samples)
    # we want ceil(n/2) rows and 2 columns
    fig, axs = plt.subplots(int(len(samples.keys()) + 1) / 2, 2)
    for i, parameter in enumerate(samples.keys()):
        num_chains = len(samples["parameter"])
        # basic bounds checking to set our axs obj
        row = i if i < len(axs) else i - len(axs)
        col = 1 if i >= len(axs) else 0
        axs[row, col].set_title(parameter)
        axs[row, col].plot(
            np.transpose(samples[parameter]), label=range(num_chains)
        )
    fig.legend()
    plt.show()
    return fig, axs


def get_timeline_from_solution_with_command(
    sol: tuple[Array, Array, Array, Array],
    compartment_idx: IntEnum,
    w_idx: IntEnum,
    strain_idx: IntEnum,
    command: str,
):
    """
    A function designed to execute `command` over a `sol` object, returning a timeline after `command` is used to select a certain view of `sol`

    Possible values of `command` include:

    - a compartment title, as specified in the `compartment_idx` IntEnum. Eg:"S", "E", "I"
    - a strain title, as specified in `strain_idx` IntEnum. Eg "omicron", "delta"
    - a wane index, as specified by `w_idx`. Eg: "W0" "W1"
    - a numpy slice of a compartment title, as specified in the `compartment_idx` IntEnum. Eg: "S[:, 0, 0, :]" or "E[:, 1:3, [0,1], 1]"
    Format must include compartment title, followed by square brackets and comma separated slices.
    Do NOT include extra time dimension found in the sol object. Assume dimensionality of the compartment as in initialization.

    Parameters
    ----------
    `sol` : tuple(jnp.array)
        generally .ys object containing ODE run as described by https://docs.kidger.site/diffrax/api/solution/
        a tuple containing the ys of the ODE run.
    `compartment_idx`: IntEnum:
        an enum containing the name of each compartment and its associated compartment index,
        as initialized by the config file of the model that generated `sol`
    `w_idx`: IntEnum:
        an enum containing the name of each waning compartment and its associated compartment index,
        as initialized by the config file of the model that generated `sol`
    `strain_idx`: intEnum
        an enum containing the name of each strain and its associated strain index,
        as initialized by the config file of the model that generated `sol`
    `command`: str
        a string command of the format specified in the function description.

    Returns
    ----------
    tuple(jnp.array, str):
        a slice of the `sol` object collapsed into the first dimension of the command selected.
    eg: return.shape = sol[0].shape[0] since all first dimensions in sol are equal normally.
        label: a string with the label of the new line,
    this helps with interpretability as commands sometimes lack necessary context
    """

    def is_close(x):
        return 0 if np.isclose(x, 0.0) else x

    is_close_v = np.vectorize(is_close)
    label = command
    # plot whole compartment
    if command in compartment_idx._member_names_:
        compartment = np.array(sol[compartment_idx[command]])
    # plot infections from that strain
    elif command in strain_idx._member_names_:
        exposed = np.array(sol[compartment_idx["E"]])
        infected = np.array(sol[compartment_idx["I"]])
        compartment = (
            exposed[:, :, :, :, strain_idx[command]]
            + infected[:, :, :, :, strain_idx[command]]
        )
        label = "E + I : " + label
    # plot members of a wane compartment
    elif command in w_idx._member_names_:
        compartment = np.array(sol[compartment_idx["S"]])[
            :, :, :, :, w_idx[command]
        ]
    # plot incidence, which is the diff of the C compartment.
    elif command.lower().strip() == "incidence":
        compartment = np.array(sol[compartment_idx["C"]])
        compartment = np.sum(
            compartment, axis=tuple(range(1, compartment.ndim))
        )
        compartment = np.diff(compartment)
        compartment_daily = is_close_v(
            np.add.reduceat(compartment, np.arange(0, len(compartment), 1))
        )
        label = "E : " + label
        return compartment_daily, label
    # plot strain prevalence, proportion of all current infections by strain over time.
    elif command.lower().strip() == "strain_prevalence":
        exposed = np.array(sol[compartment_idx["E"]])
        infected = np.array(sol[compartment_idx["I"]])
        all_cur_infected = is_close_v(exposed + infected)
        strain_proportions = []
        # normalize each strain, getting its proportion of all infected at that time point
        # strains sum to 1 for each given time point
        for strain in strain_idx._member_names_:
            strain_proportions.append(
                np.nan_to_num(all_cur_infected[:, :, :, :, strain_idx[strain]])
            )
        # sum all three strains together for normalization purposes
        all_cur_infected = np.sum(all_cur_infected, axis=-1)
        dimensions_to_sum_over = tuple(range(1, strain_proportions[0].ndim))
        strain_proportions_summed = [
            np.sum(strain_proportion_timeline, axis=dimensions_to_sum_over)
            / np.sum(all_cur_infected, axis=dimensions_to_sum_over)
            for strain_proportion_timeline in strain_proportions
        ]
        labels = [str(strain) for strain in strain_idx._member_names_]
        return strain_proportions_summed, labels

    # assuming explicit compartment, will explode if passed incorrect input
    else:
        compartment_slice = command[1:].strip()
        # add an extra dimension : for time
        compartment_slice = compartment_slice[0] + ":," + compartment_slice[1:]
        try:
            compartment_slice = eval("np.s_{}".format(compartment_slice))
            compartment = np.array(
                sol[compartment_idx[command[0].upper()]][compartment_slice]
            )
        except NameError:
            print(
                "There was an error in the plotting command: {}, returning null timeline".format(
                    command
                )
            )
            print(
                "Please review `utils/get_timeline_from_solution_with_command()` documentation"
            )
            return np.zeros(sol[compartment_idx["S"]].shape[0]), "Error"
    dimensions_to_sum_over = tuple(range(1, compartment.ndim))
    # compartment = np.nan_to_num(compartment, copy=True, nan=0.0)
    return is_close_v(np.sum(compartment, axis=dimensions_to_sum_over)), label


def get_var_proportions(inferer, solution):
    """
    Calculate _daily_ variant proportions based on a simulation run.

    Parameters
    ----------
    `inferer` : AbstractParameters
        an AbstractParameters (e.g., MechanisticInferer or StaticValueParameters) that
        is used to produce `solution`.
    `solution`: tuple(jnp.array)
        solution object that comes out from an ODE run (specifically through
        `diffrax.diffeqsolve`)

    Returns
    ----------
    jnp.array:
        an array of strain prevalence by the shape of (num_days, NUM_STRAINS)
    """
    strain_incidence = jnp.sum(
        solution.ys[inferer.config.COMPARTMENT_IDX.C],
        axis=(
            inferer.config.I_AXIS_IDX.age + 1,  # offset for day dimension
            inferer.config.I_AXIS_IDX.hist + 1,
            inferer.config.I_AXIS_IDX.vax + 1,
        ),
    )
    strain_incidence = jnp.diff(strain_incidence, axis=0)
    sim_vars = strain_incidence / jnp.sum(strain_incidence, axis=-1)[:, None]
    return sim_vars


def get_seroprevalence(inferer, solution):
    """
    Calculate the seroprevalence (more precisely the cumulative attack rate) based on
    a simulation run.

    Parameters
    ----------
    `inferer` : AbstractParameters
        an AbstractParameters (e.g., MechanisticInferer or StaticValueParameters) that
        is used to produce `solution`.
    `solution`: tuple(jnp.array)
        solution object that comes out from an ODE run (specifically through
        `diffrax.diffeqsolve`)

    Returns
    ----------
    jnp.array:
        an array of seroprevalence by the shape of (num_days, NUM_AGE_GROUPS)
    """
    never_infected = jnp.sum(
        solution.ys[inferer.config.COMPARTMENT_IDX.S][:, :, 0, :, :],
        axis=(
            # offset for day dimension, un-offset by infection history
            inferer.config.S_AXIS_IDX.vax,
            inferer.config.S_AXIS_IDX.wane,
        ),
    )
    sim_sero = 1 - never_infected / inferer.config.POPULATION
    return sim_sero


def get_foi_suscept(p, force_of_infection):
    """
    Calculate the force of infections experienced by the susceptibles, _after_
    factoring their immunity.

    Parameters
    ----------
    `p` : Parameters
        a Parameters object which is a spoofed dictionary for easy referencing,
        which is an output of `.get_parameters()` from AbstractParameter.
    `force_of_infection`: jnp.array
        an array of (NUM_AGE_GROUPS, NUM_STRAINS) that quantifies the force of
        infection experienced by age group by strain.

    Returns
    ----------
    jnp.array:
        an array of immunity protection by the shape of (NUM_STRAINS, num_days,
        NUM_AGE_GROUPS)
    """
    foi_suscept = []
    for strain in range(p.NUM_STRAINS):
        force_of_infection_strain = force_of_infection[
            :, strain
        ]  # (num_age_groups,)

        crossimmunity_matrix = p.CROSSIMMUNITY_MATRIX[strain, :]
        vax_efficacy_strain = p.VACCINE_EFF_MATRIX[strain, :]
        initial_immunity = 1 - jnp.einsum(
            "j, k",
            1 - crossimmunity_matrix,
            1 - vax_efficacy_strain,
        )
        # renormalize the waning curve to have minimum of `final_immunity` after full waning
        # and maximum of `initial_immunity` right after recovery
        final_immunity = jnp.zeros(shape=initial_immunity.shape)
        final_immunity = final_immunity.at[
            all_immune_states_with(strain, p.NUM_STRAINS), :
        ].set(p.MIN_HOMOLOGOUS_IMMUNITY)
        waned_immunity_baseline = jnp.einsum(
            "jk,l",
            initial_immunity,
            p.WANING_PROTECTIONS,
        )
        # find the lower bound of immunity for a homologous exposure against this challenging strain
        waned_immunity_min = (1 - waned_immunity_baseline) * final_immunity[
            :, :, jnp.newaxis
        ]
        waned_immunity = waned_immunity_baseline + waned_immunity_min
        foi_suscept_strain = jnp.einsum(
            "i, jkl", force_of_infection_strain, 1 - waned_immunity
        )
        foi_suscept.append(foi_suscept_strain)

    return foi_suscept


def get_immunity(inferer, solution):
    """
    Calculate the age-strain-specific population immunity. Specifically, the expected
    immunity of a randomly selected person of certain age towards certain strain.

    Parameters
    ----------
    `inferer` : AbstractParameters
        an AbstractParameters (e.g., MechanisticInferer or StaticValueParameters) that
        is used to produce `solution`.
    `solution`: tuple(jnp.array)
        solution object that comes out from an ODE run (specifically through
        `diffrax.diffeqsolve`)

    Returns
    ----------
    jnp.array:
        an array of immunity protection by the shape of (NUM_STRAINS, num_days,
        NUM_AGE_GROUPS)
    """
    p = Parameters(inferer.get_parameters())
    foi_suscept = get_foi_suscept(
        p, jnp.ones((p.NUM_AGE_GROUPS, p.NUM_STRAINS))
    )
    immunity_strain = [
        [
            1
            - jnp.sum(foi_suscept[strain] * s, axis=(1, 2, 3))
            / jnp.sum(s, axis=(1, 2, 3))
            for s in solution.ys[inferer.config.COMPARTMENT_IDX.S]
        ]
        for strain in range(p.NUM_STRAINS)
    ]

    immunity_strain = jnp.array(immunity_strain)
    return immunity_strain


def get_vaccination_rates(inferer, num_day):
    """
    Calculate _daily_ vaccination rates over the course of `num_day`.

    Parameters
    ----------
    `inferer` : AbstractParameters
        an AbstractParameters (e.g., MechanisticInferer or StaticValueParameters) that
        is used to produce `solution`.
    `num_day`: int
        number of simulation days

    Returns
    ----------
    list:
        list of `num_day` vaccination rates arrays, each by the shape of (NUM_AGE_GROUPS,
        MAX_VACCINATION_COUNT + 1)
    """
    return [inferer.vaccination_rate(t).tolist() for t in range(num_day)]


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# CONTACT MATRIX CODE
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


def rho(M: np.ndarray) -> np.ndarray:
    return np.max(np.real(np.linalg.eigvals(M)))


def make_two_settings_matrices(
    path_to_population_data: str,
    path_to_settings_data: str,
    region: str = "United States",
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    For a single region, read the two column (age, population counts) population
    csv (up to age 85) then read the 85 column interaction settings csvs by
    setting (four files) and combine them into an aggregate 85 x 85 matrix
    (twice, one for school setting and another for "other")

    Parameters
    ----------
    path_to_population_data : str
        The path to the folder that has the population size for each age
    path_to_settings_data : str
        The path to age-by-age contacts by settings
    region : str
        The FIPS region to create the 2 settings matrices

    Returns
    -------
    tuple
        A tuple containing an 85x85 school settings contact matrix, an 85x85
        other (household + community + work) contact matrix, and the population
        in each of the 85 ages groupings
    """
    # Define the settings in the interactions by settings data
    interaction_settings = ["school", "household", "community", "work"]
    # Get all the interactions by settings data
    all_settings_data_files = glob.glob(f"{path_to_settings_data}/*.csv")
    # Index where the location of the setting is after splitting by "_"
    # NB: A file name looks like United_States_country_level_F_school_setting_85.csv
    setting_index = -3
    # Collect and unpack all the settings data files, as well as the age
    # distribution, by the region inputted
    if region != "United_States":
        region_data_file = (
            "United_States_subnational_" + region + "_age_distribution_85.csv"
        )
        settings_data_dict = dict(
            [
                (
                    elt.split("_")[setting_index],
                    np.loadtxt(
                        elt, delimiter=",", dtype=np.float64, skiprows=0
                    ),
                )
                for elt in all_settings_data_files
                if region in elt
            ]
        )
    else:
        region_data_file = (
            "United_States_country_level_age_distribution_85.csv"
        )
        settings_data_dict = dict(
            [
                (
                    elt.split("_")[setting_index],
                    np.loadtxt(
                        elt, delimiter=",", dtype=np.float64, skiprows=0
                    ),
                )
                for elt in all_settings_data_files
                if ("country" in elt) and ("_F_" in elt)
            ]
        )
    # Make sure region_data_file exists
    assert os.path.exists(
        path_to_population_data + "/" + region_data_file
    ), f"The {region} file does not exist"
    # Load territory data
    region_data = pd.read_csv(
        path_to_population_data + "/" + region_data_file,
        header=None,
        names=["Age", "Population"],
    )
    # Create the empty base School and Other contact matrices
    sch_CM = np.zeros((region_data.shape[0], region_data.shape[0]))
    oth_CM = np.zeros((region_data.shape[0], region_data.shape[0]))
    # Iterate through the interaction settings, assembling the School and Other
    # contact matrices
    for setting in interaction_settings:
        if setting == "school":
            sch_CM = settings_data_dict[setting]
        # else:
        oth_CM += settings_data_dict[setting]
    return (sch_CM, oth_CM, region_data)


def create_age_grouped_CM(
    region_data: pd.DataFrame,
    setting_CM: np.ndarray,
    num_age_groups: int,
    minimum_age: int,
    age_limits,
) -> tuple[np.ndarray, list[float]]:
    """
    Parameters
    ----------
    region_data : pd.DataFrame
        A two column dataframe with the FIPS region's 85 ages and their
        population sizes
    setting_CM : np.ndarray
        An 85x85 contact matrix for a given setting (either school or other)

    Returns
    -------
    tuple
        A tuple containing an age-grouped
        (dc.NUM_AGE_GROUPS x dc.NUM_AGE_GROUPS) contact matrix and a list with
        the proportion of that FIPS region's population in each of those
        dc.NUM_AGE_GROUPS age groups
    """
    # Check if the received setting (all ages) matrix is square
    assert (
        setting_CM.shape[0] == setting_CM.shape[1]
    ), "Baseline contact matrix is not square."
    # Check if the dc.MINIMUM_AGE is proper
    assert 0 <= minimum_age < 84, "Please correct the value of the minimum age"
    # Check if the dc.MINIMUM_AGE is an int
    assert isinstance(
        minimum_age, int
    ), "Please make sure the minimum age is an int"
    # Check to see if the age limits specified are ordered properly
    assert age_limits[-1] < 84, "The entered age limits are not compatible"
    # Check if the upper bound of the age limits is greater than the lower bound
    assert (
        age_limits[0] < age_limits[1] + 1
    ), "The bounds for the age limits are not proper"
    # Create new age groups from the age limits, e.g. if [18,66], <18,18-64,65+
    age_groups = generate_yearly_age_bins_from_limits(age_limits)
    grouped_CM = np.empty(
        (num_age_groups, num_age_groups), dtype=setting_CM.dtype
    )
    # Get the population data to be used for proportions in the
    pop_proportions = region_data["Population"].div(
        region_data["Population"].sum()
    )
    # Fill in the age-grouped contact matrix
    for i, grp_out in enumerate(age_groups):
        for j, grp_in in enumerate(age_groups):
            cm_slice = setting_CM[np.ix_(grp_out, grp_in)]
            pop_prop_slice = pop_proportions[pd.Index(grp_out)] / np.sum(
                pop_proportions[pd.Index(grp_out)]
            )
            pop_prop_slice = np.reshape(pop_prop_slice.to_numpy(), (-1, 1))
            grouped_CM[i, j] = np.sum(pop_prop_slice * cm_slice)
    # Population proportions in each age group
    N_age = [np.sum(pop_proportions[pd.Index(group)]) for group in age_groups]
    return (grouped_CM, N_age)


def load_demographic_data(
    demographics_path,
    regions,
    num_age_groups,
    minimum_age,
    age_limits,
) -> dict[str, dict[str, np.ndarray]]:
    """
    Loads demography data for the specified FIPS regions, contact mixing data sourced from:
    https://github.com/mobs-lab/mixing-patterns

    Returns
    -------
    demographic_data : dict
        A dictionary of FIPS regions, with 2 age-grouped contact matrices by
        setting (school and other, where other is comprised of work, household,
        and community settings data) and data on the population of the region
        by age group
    """
    # Get the paths to the 3 files we need
    path_to_settings_data = demographics_path + "contact_matrices"
    path_to_population_data = (
        demographics_path + "population_rescaled_age_distributions"
    )
    # Check if the paths to the files exists
    assert os.path.exists(
        demographics_path
    ), f"The base path {demographics_path} does not exist as it should"
    assert os.path.exists(
        path_to_settings_data
    ), "The path to the contact matrices does not exist as it should"
    assert os.path.exists(
        path_to_population_data
    ), "The path to population-rescaled age distributions does not exist as it should"
    # Create an empty dictionary for the demographic data
    demographic_data = dict([(r, "") for r in regions])
    # Create contact matrices
    for r in regions:
        try:
            # e.g., if territory is "North Carolina", pass it as "North_Carolina"
            if len(r.split()) > 1:
                region = "_".join(r.split())
            else:
                region = r
            # Get base school and other contact matrices (for all 85 ages) and
            # the populations of each of these ages
            sch_CM_all, avg_CM_all, region_data = make_two_settings_matrices(
                path_to_population_data,
                path_to_settings_data,
                region,
            )
            # Create the age-grouped school setting contact
            sch_CM, N_age_sch = create_age_grouped_CM(
                region_data,
                sch_CM_all,
                num_age_groups,
                minimum_age,
                age_limits,
            )
            # Create the age-grouped other an average setting contact (average being all settings combined, including school)
            avg_CM, N_age_oth = create_age_grouped_CM(
                region_data,
                avg_CM_all,
                num_age_groups,
                minimum_age,
                age_limits,
            )
            # Save one of the two N_ages (they are the same) in a new N_age var
            N_age = N_age_sch
            # Rescale contact matrices by leading eigenvalue
            avg_CM = avg_CM / rho(avg_CM)
            sch_CM = sch_CM / rho(sch_CM)
            # Transform Other cm with the new age limits [NB: to transpose?]
            region_demographic_data_dict = {
                "sch_CM": sch_CM.T,
                "avg_CM": avg_CM.T,
                "N_age": np.array(N_age),
                "N": np.array(np.sum(N_age)),
            }
            demographic_data[r] = region_demographic_data_dict
        except Exception as e:
            print(
                f"Something went wrong with {region} and produced the following error:\n\t{e}"
            )
            raise e
    return demographic_data


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# SPOOFING CODE
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


class Parameters(object):
    """A dummy container that converts a dictionary into attributes."""

    def __init__(self, dict: dict):
        self.__dict__ = dict


class dual_logger_out(object):
    """
    a class that splits stdout, flushing its contents to a file as well as to stdout
    this is useful for Azure Batch to save logs but also see the output live on the node
    """

    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self

    def close(self):
        sys.stdout = self.stdout
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()


class dual_logger_err(object):
    """
    a class that splits stderror, flushing its contents to a file as well as to terminal if an error occurs
    this is useful for Azure Batch to save logs but also see the output live on the node
    """

    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stderr = sys.stderr
        sys.stderr = self

    def close(self):
        sys.stderr = self.stderr
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stderr.write(data)

    def flush(self):
        self.file.flush()


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# OS operations CODE
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


def find_files(
    directory: str, filename_contains: str, recursive=False
) -> list[str]:
    """searched `directory` for any files with `filename_contains`,
    optionally searched recrusively down from `directory`

    Parameters
    ----------
    directory : str
        directory, absolute or relative from which to start search
    filename_contains : str
        partial file name to search for
    recursive : bool (optional)
        whether to recursively search subfolders within `directory`

    Returns
    -------
    list[str]
        list of filenames containing `filename_contains`
    """
    # Create a pattern to match filenames containing 'postprocess' with any extension
    if recursive:
        pattern = directory + "/**/*%s*.*" % filename_contains
    else:
        pattern = directory + "/*%s*.*" % filename_contains

    # Use glob to find all matching files
    postprocess_files = glob.glob(pattern, recursive=recursive)

    return [os.path.basename(file) for file in postprocess_files]
