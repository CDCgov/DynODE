"""utility functions used within various components of initialization, inference, and interpretation."""

import datetime
import glob
import os

# importing under a different name because mypy static type hinter
# strongly dislikes the IntEnum class.
from typing import Any

import epiweeks
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pandas as pd  # type: ignore
from jax import Array

pd.options.mode.chained_assignment = None


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# INDEXING FUNCTIONS
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


def sim_day_to_date(sim_day: int, init_date: datetime.date):
    """Compute date object for given `sim_day` and `init_date`.

    Given current model's simulation day as integer and
    initialization date, returns date object representing current simulation day.

    Parameters
    ----------
    sim_day : int
        Current model simulation day where sim_day==0==init_date.

    init_date : datetime.date
        Initialization date usually found in config.INIT_DATE parameter.

    Returns
    -------
    datetime.date object representing current `sim_day`

    Examples
    --------
    >>> import datetime
    >>> init_date = datetime.date(2022, 10, 15)
    >>> sim_day_to_date(10, init_date )
    datetime.date(2022, 10, 25 )
    """
    return init_date + datetime.timedelta(days=sim_day)


def sim_day_to_epiweek(
    sim_day: int, init_date: datetime.date
) -> epiweeks.Week:
    """Calculate CDC epiweek that sim_day falls in.

    Parameters
    ----------
    sim_day : int
        Current model simulation day where sim_day==o==init_date.

    init_date : datetime.date
        Initialization date usually found in config.INIT_DATE parameter.

    Returns
    -------
    epiweeks.Week
        CDC epiweek on day sim_day

    Examples
    --------
    >>> import datetime
    >>> init_date=datetime.date(2022, 10, 15)
    >>> sim_day_to_epiweek(10, init_date )
    epiweeks.Week(year=2022, week=42)
    """
    date = sim_day_to_date(sim_day, init_date)
    epi_week = epiweeks.Week.fromdate(date)
    return epi_week


def date_to_sim_day(date: datetime.date, init_date: datetime.date):
    """Convert date object to simulation days using init_date as reference point.

    Parameters
    ----------
    date : datetime.date
        Date being converted into integer simulation days.

    init_date : datetime.date
        Initialization date usually found in config.INIT_DATE parameter.

    Returns
    -------
    int
    how many days have passed since `init _date`

    Examples
    --------
    >>> import datetime
    >>> init_date=datetime.date(2022, 10, 15)
    >>> date=datetime.date(2022, 11, 5)
    >>> date_to_sim_day(date, init_date)
    21
    """
    return (date - init_date).days


def date_to_epi_week(date: datetime.date):
    """Convert a date object to CDC epi week.

    Parameters
    ----------
    sim_day : datetime.date
        Date to be converted to a simulation day.

    Returns
    -------
    epiweeks.Week
        The epi_week that `date` falls in.
    """
    epi_week = epiweeks.Week.fromdate(date)
    return epi_week


def new_immune_state(current_state: int, exposed_strain: int) -> int:
    """Determine a new immune state after applying an exposing strain to an immune state.

    Uses bitwise OR given the current state and the exposing strain.

    Parameters
    ----------
    current_state : int
        Int representing the current state of the
        individual or group being exposed to a strain.
    exposed_strain : int
        Int representing the strain exposed to the
        individuals in state `current_state`.
        expects that `0 <= exposed_strain <= num_strains - 1`.

    Returns
    -------
    int
        Individual or population's new immune state after exposure and recovery
        from `exposed_strain`.

    Examples
    --------
    num_strains = 2, possible states are:
    00(no exposure), 1(exposed to strain 0 only), 2(exposed to strain 1 only),
    3(exposed to both)

    >>> new_immune_state(current_state = 0, exposed_strain = 0)
    1 #no previous exposure, now exposed to strain 0
    >>> new_immune_state(0, 1)
    2 #no previous exposure, now exposed to strain 1
    >>> new_immune_state(1, 0)
    1 #exposed to strain 0 already, no change in state
    >>> new_immune_state(2, 1)
    2 #exposed to strain 1 already, no change in state
    >>> new_immune_state(1, 1)
    3 #exposed to strain 0 previously, now exposed to both
    >>> new_immune_state(2, 0)
    3 #exposed to strain 1 previously, now exposed to both
    >>> new_immune_state(3, 0)
    3 #exposed to both already, no change in state
    >>> new_immune_state(3, 1)
    3 #exposed to both already, no change in state
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
    else:  # being passed jax.ArrayLike
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
    """Determine all immune states which contain an exposure to `strain`.

    Parameters
    ----------
    strain : int
        Int representing the exposed-to strain,
        expects that `0 <= strain <= num_strains - 1`.
    num_strains : int
        Number of strains in the model.

    Returns
    -------
    list[int]
        all immune states that include previous exposure to `strain`

    Examples
    --------
    in a simple model where num_strains = 2
    Reminder:
    state = 0 (no exposure),
    state = 1/2 (exposure to strain 0/1 respectively),
    state = 3 (exposed to both)

    >>> all_immune_states_with(strain = 0, num_strains = 2)
    [1, 3]
    >>> all_immune_states_with(strain = 1, num_strains = 2)
    [2, 3]
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
    """Determine all immune states which do not contain an exposure to `strain`.

    Parameters
    ----------
    strain : int
        Int representing the NOT exposed to strain,
        expects that `0 <= strain <= num_strains - 1`.
    num_strains : int
        Number of strains in the model.

    Returns
    -------
    list[int] representing all immune states that
    do not include previous exposure to `strain`

    Examples
    --------
    in a simple model where num_strains = 2.
    Reminder:
    state = 0 (no exposure),
    state = 1/2 (exposure to strain 0/1 respectively),
    state = 3 (exposed to both)

    >>> all_immune_states_with(strain = 0, num_strains = 2)
    [0, 2]
    >>> all_immune_states_with(strain = 1, num_strains = 2)
    [0, 1]
    """
    all_states = list(range(2**num_strains))
    states_with_strain = all_immune_states_with(strain, num_strains)
    # return set difference of all states and states including strain
    return list(set(all_states) - set(states_with_strain))


def get_strains_exposed_to(state: int, num_strains: int):
    """Unpack all strain exposures an immune state was exposed to.

    Says nothing of the order at which an individual was exposed to strains.

    Parameters
    ----------
    state : int
        The state a given individual is in, as dicated by a single or series of
        exposures to strains. State dynamics determined by `new_immune_state()`.
    num_strains : int
        The total number of strains in the model,
        used to determin total size of state space.

    Returns
    -------
    list[int]
        strains the individual in `state` was exposed to.
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
    """Merge two strain definitions together.

    Parameters
    ----------
    from_strain : int
        The strain index representing the strain being collapsed,
        whos references will be rerouted.
    to_strain : int
        The strain index representing the strain being joined with
        to_strain, typically the ancestral or 0 index.
    num_strains : int
        Number of strains in the model, constrains immune state space.

    Returns
    -------
    tuple(dict[int,int], dict[int,int])
        First dict[int,int] maps from immune state -> immune state before and
        after `from_strain` is combined with `to_strain` for all states.

        Second dict[int,int] maps from strain idx -> strain idx
        before and after`from_strain` is combined with `to_strain` for all strains.

    Examples
    --------
    In a basic 2 strain model you have the following immune states:
    0-> no exposure, 1 -> strain 0 exposure,
    2-> strain 1 exposure, 3-> exposure to both

    >>> combine_strains(from_strain = 1, to_strain = 0, num_strains = 2)
    ({0:0, 1:1, 2:1, 3:1}, {0:0, 1:0}),
    # immune state space becomes binary.
    # both strain 0 and 1 now route to strain 0
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
    """Merge two or more strain definitions together within a compartment.

    Combines the state dimensions and optionally the strain dimension if
    `strain_axis=True`.

    Parameters
    ----------
    compartment : np.ndarray
        The compartment being changed, must be four dimensional
        with immune state in the `state_dim` dimension and
        strain (if applicable) in the `strain_dim` dimension.
    state_mapping : dict[int:int]
        A mapping of pre-combine state to post-combine state,
        as generated by `combined_strains_mapping()`,
        must cover all states found in `compartment[state_dim]`.
    strain_mapping : dict[int:int]
        A mapping of pre-combine strain to post-combine strain,
        as generated by `combined_strains_mapping()`,
        must cover all strains found in `compartment[strain_dim]`.
    num_strains : int
        Number of strains in the model.
    state_dim : int
        Which dimension in `compartment` immune state is found in, default 1.
    strain_dim : int
        Which dimension in `compartment` strain num is found in, if applicable,
        default 3.
    strain_axis : bool
        Whether or not `compartment` includes a strain axis
        in `strain_dim`. Not all compartments track `strain`.

    Returns
    -------
    np.ndarray:
        A modified copy of `compartment` with all immune states and
        strains combined according to `state_mapping` and `strain_mapping`
    """
    # begin with a copy of the compartment in all zeros
    strain_combined_compartment = np.zeros(compartment.shape)
    # next update the immune states according to the state_mapping dict
    for immune_state in range(2**num_strains):
        # after strain combining immune_state moves to `new_state`
        new_state = state_mapping[immune_state]
        # += because multiple `immune_states` can flow into one `new_state`
        # use swapaxis to grab an arbitrary dimension of the array, ignoring ordering bugs
        strain_combined_compartment.swapaxes(0, state_dim)[new_state] += (
            compartment.swapaxes(0, state_dim)[immune_state]
        )
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


def strain_interaction_to_cross_immunity(
    num_strains: int, strain_interactions: np.ndarray
) -> Array:
    """
    Convert a strain interaction matrix to a cross-immunity matrix.

    Parameters
    ----------
    num_strains : int
        Number of strains in the model.
    strain_interactions : np.ndarray
        Matrix (num_strains, num_strains) representing
        relative immunity from one strain to another.
        `strain_interactions[i][j] = 1.0` states
        full immunity from challenging strain `i` after
        recovery from strain `j`.

    Returns
    -------
    jax.Array
        Matrix (num_strains, 2**num_strains) representing immunity
        for all immune history permutations against a challenging strain.

    Notes
    -----
    Relative immunity does not account for waning.
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


def flatten_list_parameters(
    samples: dict[str, np.ndarray | Array],
) -> dict[str, np.ndarray | Array]:
    """
    Flatten plated parameters into separate keys in the samples dictionary.

    Parameters
    ----------
    samples : dict[str, np.ndarray | Array]
        Dictionary with parameter names as keys and sample
        arrays as values. Arrays may have shape MxNxP for P independent draws.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary with plated parameters split into
        separate keys. Each new key has arrays of shape MxN.

    Notes
    -----
    If no plated parameters are present, returns a copy of the dictionary.
    """
    return_dict = {}
    for key, value in samples.items():
        if isinstance(value, (np.ndarray, Array)) and value.ndim > 2:
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
    """
    Drop keys from a dictionary if they contain a specified substring.

    Parameters
    ----------
    dct : dict[str, Any]
        Dictionary with string keys.
    drop_s : str
        Substring to check for in keys.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys containing `drop_s` removed.
    """
    keys_to_drop = [key for key in dct.keys() if drop_s in key]
    for key in keys_to_drop:
        del dct[key]
    return dct


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# DEMOGRAPHICS CODE
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


def generate_yearly_age_bins_from_limits(age_limits: list) -> list[list[int]]:
    """Generate age bins up to 85 years old exclusive based on age limits.

    Parameters
    ----------
    age_limits : list[int]
        Boundaries of each age bin. The last bin is implicitly up to 85.

    Returns
    -------
    list[list[int]]
        List of lists containing integer years within each age bin.

    Examples
    --------
    >>> generate_yearly_age_bins_from_limits([0, 5, 10, ... 80])
    [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9],... [80, 81, 82, 83, 84]]
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
    """Load normalized proportions of each age bin for given regions.

    Parameters
    ----------
    path : str
        Path to the demographic-data folder.
    regions : list(str)
        List of FIPS regions
    age_limits : list(int)
        Age limits for each bin; values are exclusive upper bounds.
        Max tracked age is enforced at 84 inclusive. All
        populations older than 84 are counted as 84 years old.

    Returns
    -------
    demographic_data : dict[str, np.ndarray]
        A dictionary maping FIPS region supplied in `regions`
        to an array of length `len(age_limits)` representing
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
# CONTACT MATRIX CODE
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


def make_two_settings_matrices(
    path_to_population_data: str,
    path_to_settings_data: str,
    region: str = "United States",
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load and parse settings contact matricies for a given region.

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
    """Load a contact matrix and group it into age bins.

    Parameters
    ----------
    region_data : pd.DataFrame
        A two column dataframe with the FIPS region's 85 ages and their
        population sizes
    setting_CM : np.ndarray
        An 85x85 contact matrix for a given setting (either school or other)
    num_age_groups : int
        number of age bins.
    minimum_age : int
        lowest possible tracked age in years.
    age_limits : list[int]
        Age limit for each age bin in the model, beginning with minimum age,
        values are exclusive in upper bound. so [0,18] means 0-17, 18+.

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
    """Load demography data for the specified FIPS regions.

    Contact mixing data sourced often from:
    https://github.com/mobs-lab/mixing-patterns

    Parameters
    ----------
    demographics_path : str
        path to demographic data directory, contains "contact_matrices" and
        "population_rescaled_age_distributions" directories.
    regions : list[str]
        list of FIPS regions to load.
    num_age_groups : int
        number of age bins.
    minimum_age : int
        lowest possible tracked age in years.
    age_limits : list[int]
        Age limit for each age bin in the model, beginning with minimum age,
        values are exclusive in upper bound. so [0,18] means 0-17, 18+.

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
            avg_CM = avg_CM / np.max(np.real(np.linalg.eigvals(avg_CM)))
            sch_CM = sch_CM / np.max(np.real(np.linalg.eigvals(sch_CM)))
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


def save_samples(samples: dict[str, Array], save_path: str, indent=None):
    """
    Save model samples to `save_path`, with JSON serializability.

    Parameters
    ----------
    samples : dict[str, Array]
        Dictionary with str keys each containing a jax array of samples or
        posteriors, often returned by MCMC.get_samples()
    save_path : str
        path, relative or absolute to save json to
    indent : Optional[int]
        optional indent spaces for pretty-printing of json,
        None is most compact and least readable, by default None
    """
    import json

    # convert np arrays to lists
    s = {param: samples[param].tolist() for param in samples.keys()}
    json.dump(s, open(save_path, "w"), indent=indent)


def identify_distribution_indexes(
    parameters: dict[str, Any],
) -> dict[str, dict[str, str | tuple | None]]:
    """Identify the locations and site names of numpyro samples.

    The inverse of `sample_if_distribution()`, identifies which parameters
    are numpyro distributions and returns a mapping between the sample site
    names and its actual parameter name and index.

    Parameters
    ----------
    parameters : dict[str, Any]
        A dictionary containing keys of different parameter
        names and values of any type.

    Returns
    -------
    dict[str, dict[str, str | tuple[int] | None]]
        A dictionary mapping the sample name to the dict key within `parameters`.
        If the sampled parameter is within a larger list, returns a tuple of indexes as well,
        otherwise None.

        - key: `str`
            Sampled parameter name as produced by `sample_if_distribution()`.
        - value: `dict[str, str | tuple | None]`
            "sample_name" maps to key within `parameters` and "sample_idx" provides
            the indexes of the distribution if it is found in a list, otherwise None.

    Examples
    --------
    >>> import numpyro.distributions as dist
    >>> parameters = {"test": [0, dist.Normal(), 2], "example": dist.Normal()}
    >>> identify_distribution_indexes(parameters)
    {'test_1': {'sample_name': 'test', 'sample_idx': (1,)},
    'example': {'sample_name': 'example', 'sample_idx': None}}
    """

    def get_index(indexes):
        return tuple(indexes)

    index_locations: dict[str, dict[str, str | tuple | None]] = {}
    for key, param in parameters.items():
        # if distribution, it does not have an index, so None
        if issubclass(type(param), dist.Distribution):
            index_locations[key] = {"sample_name": key, "sample_idx": None}
        # if list, check for distributions within and mark their indexes
        elif isinstance(param, (np.ndarray, list)):
            param = np.array(param)  # cast np.array so we get .shape
            flat_param = np.ravel(param)  # Flatten the parameter array
            # check for distributions inside of the flattened parameter list
            if any(
                [
                    issubclass(type(param_lst), dist.Distribution)
                    for param_lst in flat_param
                ]
            ):
                dim_idxs = np.unravel_index(
                    np.arange(flat_param.size), param.shape
                )
                for i, param_lst in enumerate(flat_param):
                    if issubclass(type(param_lst), dist.Distribution):
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
