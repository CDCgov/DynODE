import datetime
import glob
import os

import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd

pd.options.mode.chained_assignment = None


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# SAMPLING FUNCTIONS
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
def sample_r0():
    """sample r0 for each a single strain according to an exponential distribution with rate 1.0

    Returns
    ----------
    numpro.sample object with name `r0` containing a deterministic object 1+exp(1.0)
    """

    excess_r0 = numpyro.sample(
        "excess_r0", numpyro.distributions.Exponential(1.0)
    )
    r0 = numpyro.deterministic("r0", 1 + excess_r0)
    return r0


def sample_waning_protections(waning_protect_means):
    """Sample a waning rate for each of the waning comparments according to an exponential distribution
    with rate equal to 1 / waning_protect_means

    Parameters
    ----------
    waning_protect_means: list(int)
        list of mean waning protection for each waning compartment
        len(waning_protect_means) = # of waning compartments in your model.
    """
    waning_rates = []
    for i, sample_mean in enumerate(waning_protect_means):
        waning_protection = numpyro.sample(
            "waning_protection_" + str(i), dist.Exponential(1 / sample_mean)
        )
        waning_rates.append(waning_protection)
    return waning_rates


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# INDEXING FUNCTIONS
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


def new_immune_state(current_state, exposed_strain, num_strains):
    """a method using BITWISE OR to determine a new immune state position given
    current state and the exposing strain

    Parameters
    ----------
    current_state: int
        int representing the current state of the individual or group being exposed to a strain
    exposed_strain: int
        int representing the strain exposed to the individuals in state `current_state`
        expects that `0 <= exposed_strain <= num_strains - 1`
    num_strains: int
        number of strains in the model
    Examples
    ----------
    num_strains = 2, possible states are:
    00(no exposure), 1(exposed to strain 0 only), 2(exposed to strain 1 only), 3(exposed to both)

    exposed strains are transformed by this function into:
    exposing strain = 0 -> represented by 01. exposed_strain = 1 -> represented by 10.

    current state | exposed strain -> new state -> int(new state)
    00 | 01 -> 01 -> 1 no previous exposure, now exposed to strain 0.
    01 | 01 -> 01 -> 1 exposed to strain 0 already, no change in state
    01 | 10 -> 11 -> 3 exposed to strain 0 prev, now exposed to both
    10 | 01 -> 11 -> 3 exposed to strain 1 prev, now exposed to both
    10 | 10 -> 10 -> 2 exposed to strain 1 already, no change in state
    11 | 01 -> 11 -> 3 exposed to both already, no change in state
    11 | 10 -> 11 -> 3 exposed to both already, no change in state
    """
    assert (
        exposed_strain >= 0 and exposed_strain <= num_strains - 1
    ), "invalid exposed_strain val"
    assert current_state < 2**num_strains, "invalid current state"
    # represent current state as bit string, ex: state = 3 & num_strains = 2 -> binary = '11'
    current_state_binary = format(current_state, "b")

    # represent exposing strain as an indiciator bit string. ex: exposing_strain = 1 -> binary = 10
    exposing_strain_binary = ["0"] * num_strains
    exposing_strain_binary[-(exposed_strain + 1)] = "1"
    exposing_strain_binary = "".join(exposing_strain_binary)

    # we now have
    new_state = format(
        int(current_state_binary, 2) | int(exposing_strain_binary, 2), "b"
    )
    return int(new_state, 2)


def all_immune_states_with(strain, num_strains):
    """
    a function returning all of the immune states which contain an exposure to `strain`

    Parameters
    ----------
    strain: int
        int representing the strain that the returns states are exposed to
        expects that `0 <= strain <= num_strains - 1`
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
    strain_binary = ["0"] * num_strains
    strain_binary[-(strain + 1)] = "1"
    strain_binary = "".join(strain_binary)
    # a state contains the strain being filtered if the bitwise AND produces a non-zero value
    filtered_states = [
        int(binary, 2)
        for binary in binary_array
        if (int(binary, 2) & int(strain_binary, 2)) > 0
    ]
    return filtered_states


def all_immune_states_without(strain, num_strains):
    """
    function returning all of the immune states which DO NOT contain an exposure to `strain`

    Parameters
    ----------
    strain: int
        int representing the strain that the returns states are NOT exposed to
        expects that `0 <= strain <= num_strains - 1`
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
    all_immune_states_with(0, 2) -> [0, 2]
    all_immune_states_with(1, 2) -> [0, 1]
    """
    all_states = list(range(2**num_strains))
    states_with_strain = all_immune_states_with(strain, num_strains)
    # return set difference of all states and states including strain
    return set(all_states) - set(states_with_strain)


def find_age_bin(age, age_limits):
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


def find_vax_bin(vax_shots, max_doses):
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
    The index of the vax bin, assuming 0 is 0 vaccinations and max_doses = any vaccinations equal to or more than max_doses
    """
    return min(vax_shots, max_doses)


def convert_hist(strains, STRAIN_IDX, num_strains):
    """
    a function that transforms a comma separated list of strains and transform them into an immune history state.
    num_strains and STRAIN_IDX are often initalized in configuration files and may include less strains than those included in
    `strains`. Any unrecognized strain strings inside of `strains` do not contiribute to the returned state.

    Parameters
    ----------
    strains: str
        a comma separated string of each exposed strain, order does not matter, capitalization does not matter.
    STRAIN_IDX: intEnum
        an enum containing the name of each strain and its associated strain index, as initialized by ConfigBase.
    num_strains:
        the number of _tracked_ strains in the model. This may be less than or equal to the unique number of strains in `strains`

    """
    state = 0
    for strain in filter(None, strains.split(",")):
        if strain.lower() in STRAIN_IDX._member_map_:
            strain_idx = STRAIN_IDX[strain.lower()]
        else:
            strain_idx = 0
        state = new_immune_state(state, strain_idx, num_strains)
    return state


def convert_strain(strain, STRAIN_IDX):
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


def find_waning_compartment(TSLIE, waning_times):
    """
    Given a TSLIE (time since last immunogenetic event) in days, returns the waning compartment index of the event.

    Parameters
    ----------
    TSLIE: int
        the number of days since the initialization of the model that the immunogenetic event occured (this could be vaccination or infection).
    waning_times: list(int)
        the number of days an individual stays in each waning compartment, ending in zero as the last compartment does not wane.
    """
    current_bin = 0
    for wane_time in waning_times:
        if TSLIE - wane_time < 0:
            return current_bin
        else:
            TSLIE -= wane_time
            current_bin += 1
    # last compartment waning_time = 0, shifts us 1 extra bin, shift back in this edge case.
    return current_bin - 1


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# DEMOGRAPHICS CODE
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


def generate_yearly_age_bins_from_limits(age_limits):
    """
    given age limits, generates age bins with each year contained in that bin up to 85 years old exclusvive
    for example given age_limits = [0, 5, 10, 15 ... 80]
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
    path,
    regions,
    age_limits,
):
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

    demographic_data = dict([(r, "") for r in regions])
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


def prep_serology_data(
    path, num_historical_strains, historical_time_breakpoints
):
    """
    DEPRECATED: USE ABM INFORMED INITIALIZATION ROUTINES

    reads serology data from path, filters to only USA site,
    filters Date Ranges from Sep 2020 - Feb 2022,
    calculates monotonically increasing rates of change (to combat sero-reversion from the assay),
    and converts string dates to datetime.dt.date objects. Then interpolates all time spans into individual days.

    TODO: change method of combatting sero-reversion to one outlined here:
    https://www.nature.com/articles/s41467-023-37944-5

    Parameters
    ----------
    path: str
        relative path to serology data sourced from
        https://data.cdc.gov/Laboratory-Surveillance/Nationwide-Commercial-Laboratory-Seroprevalence-Su/d2tw-32xv

    num_historical_strains: int
        the number of historical strains to be used in the "strain_select" column of the output.
        most recent strain will always be placed as value num_historical_strains - 1. While oldest at 0.

    historical_time_breakpoints: list[datetime.date]
        list of datetime.date breakpoints on which an older strain transitions to a newer one.
        for example, omicron took off on (2021, 11, 19), meaning anything before that date is delta, on or after is omicron.


    Returns
    ----------
    serology table containing the following additional columns:
        `collection_start` = assay collection start date \n
        `collection_end` = assay collection end date \n
        `age0_age1_diff` = difference in `Rate (%) [Anti-N, age1-age2 Years Prevalence]` from current and previous collection.
        enforced to be positive or 0 to combat sero-reversion. Columns repeats for age bins [0-17, 18-49, 50-64, 65+] \n
        `strain_select` = the strain index value for sero conversion on that day. As decided by historical_time_breakpoints and
        the `num_historical_strains`

    Modifies
    ----------
    `Rate (%) [Anti-N, age1-age2 Years Prevalence, Rounds 1-30 only]` columns to enforce monotonicity.
    """
    serology = pd.read_csv(path)
    # filter down to USA and pick a date after omicron surge to load serology from.
    serology = serology[serology["Site"] == "US"]
    dates_of_interest = pd.read_csv("data/dates_of_interest.csv")[
        "date_name"
    ].values
    # pick date ranges from the dates of interest list
    serology = serology[
        [
            date in dates_of_interest
            for date in serology["Date Range of Specimen Collection"]
        ]
    ]
    # focus on anti-n sero prevalence in all age groups
    columns_of_interest = [
        "Date Range of Specimen Collection",
        "Rate (%) [Anti-N, 0-17 Years Prevalence]",
        "Rate (%) [Anti-N, 18-49 Years Prevalence, Rounds 1-30 only]",
        "Rate (%) [Anti-N, 50-64 Years Prevalence, Rounds 1-30 only]",
        "Rate (%) [Anti-N, 65+ Years Prevalence, Rounds 1-30 only]",
    ]
    serology = serology[columns_of_interest]
    # enforce monotonicity to combat sero-reversion in early pandemic serology assays
    # start at index 1 in columns of interest to avoid date column
    # TODO https://www.nature.com/articles/s41467-023-37944-5 use this method for combating sero-reversion
    for diff_column in columns_of_interest[1:]:
        for idx in range(1, len(serology[diff_column])):
            serology[diff_column].iloc[idx] = max(
                serology[diff_column].iloc[idx - 1],
                serology[diff_column].iloc[idx],
            )
        serology[diff_column] = serology[diff_column] / 100.0
    # lets create datetime objects out of collection range
    years = [
        x.split(",")[-1] for x in serology["Date Range of Specimen Collection"]
    ]
    serology["collection_start"] = pd.to_datetime(
        [
            # edge case Date = Dec 27, 2021 - Jan 29, 2022 need years
            date.split("-")[0].strip() + "," + year
            if len(date.split(",")) == 2
            else date.split("-")[0].strip()
            for date, year in zip(
                serology["Date Range of Specimen Collection"], years
            )
        ],
        format="%b %d, %Y",
    )

    serology["collection_end"] = pd.to_datetime(
        [
            x.split("-")[1].strip()
            for x in serology["Date Range of Specimen Collection"]
        ],
        format="%b %d, %Y",
    )

    # transform from datetime to date obj
    serology["collection_start"] = serology["collection_start"].dt.date
    serology["collection_end"] = serology["collection_end"].dt.date
    # pick the date between collection start and end as the point estimate for date of collection
    serology["collection_date"] = [
        start + ((end - start) / 2)
        for start, end in zip(
            serology["collection_start"], serology["collection_end"]
        )
    ]
    # after we interpolate down to daily precision, rebin into waning compartments
    serology.index = pd.to_datetime(serology["collection_date"])
    serology = serology[columns_of_interest[1:]]  # filter to only int cols
    # possible reimplementation of variable waning compartment bin width
    # will probably need to return to [::-x] slicing with a variable x or something.
    serology = (
        serology.resample(
            "1d"
        ).interpolate()  # downsample to daily freq  # linear interpolate between days
        # .resample(
        #     str(waning_time) + "d", origin="end"
        # )  # resample to waning compart width
        # .max()
    )
    strain_select = (
        num_historical_strains - 1
    )  # initialize as most recent strain

    strain_select_array = [
        strain_select
        - sum(
            [
                date < pd.Timestamp(historical_breakpoint)
                for historical_breakpoint in historical_time_breakpoints
            ]
        )
        for date in serology.index
    ]

    serology["strain_select"] = strain_select_array
    # we will use the absolute change in % serology prevalence to initialize wane compartments
    serology["0_17_diff"] = serology[
        "Rate (%) [Anti-N, 0-17 Years Prevalence]"
    ].diff()
    serology["18_49_diff"] = serology[
        "Rate (%) [Anti-N, 18-49 Years Prevalence, Rounds 1-30 only]"
    ].diff()
    serology["50_64_diff"] = serology[
        "Rate (%) [Anti-N, 50-64 Years Prevalence, Rounds 1-30 only]"
    ].diff()
    serology["65_diff"] = serology[
        "Rate (%) [Anti-N, 65+ Years Prevalence, Rounds 1-30 only]"
    ].diff()

    return serology


def prep_abm_data(
    abm_population,
    max_vax_count,
    age_limits,
    waning_times,
    num_strains,
    STRAIN_IDXs,
):
    """
    A helper function called by past_immune_dist_from_abm() that takes as input a path to some abm data with schema specified by the README,
    and applies transformations to the table, adding some columns so individuals within the ABM data are able to be placed
    in the correct partial immunity bins. This includes vaccination, age binning, waning bins, and conversion of strain exposure history
    into an immune history.

    Parameters
    ----------
    abm_population: pd.Dataframe
        ABM data input with schema specified by project README.
    max_vax_count: int
        the number of doses maximum before all subsequent doses are no longer counted. ex: 2 -> 0, 1, 2+ doses (3 bins)
    age_limits: list(int)
        The age limits of your model that you wish to initialize compartments of.
        Example: for bins of 0-17, 18-49, 50-64, 65+ age_limits = [0, 18, 50, 65]
    waning_times: list(int)
        Time in days it takes for a person to wane from a waning compartment to the next level of protection.
        len(waning_times) == num_waning_compartments, ending in 0.
    num_strains: int
        number of distinct strains in your model, used to inform the `state` column in output
    STRAIN_IDX: intEnum
        an enum containing the name of each strain and its associated strain index, as initialized by ConfigBase.

    Returns
    ----------
    A pandas dataframe read in from abm_path with 4 added columns: vax_bin, age_bin, waning_compartment_bin, and state.
    The first 3 are simple transformations made to bin a domain according to the parameters of a model.
    While the last converts a list of strain exposures into a integer state representing immune history.
    """
    # replace N/A values with empty string so that convert_state() works correctly.
    abm_population["strains"] = abm_population["strains"].fillna("")
    abm_population["vax_bin"] = abm_population["num_doses"].apply(
        lambda x: find_vax_bin(x, max_vax_count)
    )
    abm_population["age_bin"] = abm_population["age"].apply(
        lambda x: find_age_bin(x, age_limits)
    )
    abm_population["waning_compartment_bin"] = abm_population["TSLIE"].apply(
        lambda x: find_waning_compartment(x, waning_times)
    )
    abm_population["state"] = abm_population["strains"].apply(
        lambda x: convert_hist(x, STRAIN_IDXs, num_strains)
    )
    return abm_population


def set_serology_timeline(num_strains):
    """
    DEPRECATED: USE ABM INFORMED INITIALIZATION ROUTINES

    a helper method which does the logic of setting historical strain breakpoint dates.

    Takes the number of strains serology data will be used to initialize, and collapses certain strains together
    if needed. Returning the number of strains which are counted individually (after collapse).
    This value may be different than num_strains if num_strains > 3, as only 3 historical timelines are supported.

    Parameters
    ----------
    num_strains: int
        number of strains the serology data is supposed to initialize for.
        Any value greater than 3 will be treated as 3 for historical initialization.

    Returns
    -----------
    The number of historical strains to be loaded as an int.
    an array of datetime.dates representing the breakpoints between each historical date.

    Example
    ----------
    if you wish to initialize omicron, delta, and alpha strains. Num strains must be set to 3 or higher
    will return (3, [datetime.date(2021, 6, 25), datetime.date(2021, 11, 19)])
    with each date representing the date at which alpha -> delta and then delta -> omicron

    """
    # number of strains alloted for in the serological data, for covid this is omicron, delta, and alpha
    # if model only allows for 2 or 1 strain we need to collapse delta and alpha waves together
    num_historical_strains = 3 if num_strains >= 3 else num_strains
    # breakpoints for each historical strain, oldest first, alpha - delta, delta - omicron
    omicron_date = datetime.date(2021, 11, 19)  # as omicron took off
    delta_date = datetime.date(2021, 6, 25)  # as the delta wave took off.
    historical_time_breakpoints = [delta_date, omicron_date]
    # small modifications needed so this does not break 2 and 1 strain models
    if num_historical_strains == 1:
        # no breakpoints when only 1 historical strain
        historical_time_breakpoints = []
    elif num_historical_strains == 2:
        # if we are only looking at 2 historical strains, only take most recent breakpoint
        historical_time_breakpoints = [historical_time_breakpoints[-1]]
    assert (
        num_historical_strains <= num_strains
    ), "you are attempting to find sero data for more historical strains than total strains alloted to the model"

    assert (
        num_historical_strains == len(historical_time_breakpoints) + 1
    ), "set breakpoints for each of the historical strains you want to initialize with sero data"
    return num_historical_strains, historical_time_breakpoints


def imply_immune_history_dist_from_strains(
    strain_exposure_dist,
    num_strains,
    num_historical_strains,
    repeat_inf_rate=0.5,
):
    """
    DEPRECATED: USE ABM INFORMED INITIALIZATION ROUTINES

    takes a matrix of shape (age, strain, waning) and converts it to
    (age, immune_hist, waning). It does this by assuming the following:
    Any individuals who are infected by a single strain,
    half of those individuals will be re-infected by all incoming future strains.
    Immune hist is a integer state representing a history of all past infections.

    Parameters
    ----------
    strain_exposure_dist: np.array
        a numpy array of proportions of persons exposed to a variety of strains.
        stratified by age, strain, and waning compartment.
    num_strains: int
        number of strains for which sero data is being loaded
    repeat_inf_rate: float
        the rate at which those infected by one strain are re-infected by a strain in the future.

    Returns
    ----------
    immune_history_dist: np.array
        a numpy array representing proportions of the population in each immune state as informed by the
        strain_exposure_dist. Waning compartments and age structure are preserved. Strain dimension is
        modified to represent immune history, predicting multiple infections and more complex immune states.
    """
    return_shape = (
        strain_exposure_dist.shape[0],
        2**num_strains,
        3,  # TODO remove this and all MAGIC 0s after adding vax
        strain_exposure_dist.shape[2],
    )  # immune states equal to 2^num_strains
    immune_history_dist = np.zeros(return_shape)
    immune_states = []
    for strain in range(0, num_historical_strains):
        # fill in single strain immune state first. no repeated exposures yet.
        single_strain_state = new_immune_state(0, strain, num_strains)
        immune_history_dist[
            :, single_strain_state, 0, :  # TODO remove 0
        ] = strain_exposure_dist[:, strain, :]
        # now grab individuals from previous states and infect 1/2 of them with this strain
        multi_strain_states = []
        for prev_state in immune_states:
            multi_strain_state = new_immune_state(
                prev_state,
                strain,
                num_strains,
            )
            multi_strain_states.append(multi_strain_state)
            age_summed = np.sum(strain_exposure_dist[:, strain, :], axis=0)
            waning_compartments_with_strain = np.where(age_summed > 0)
            # TODO remove 0s
            # following for loop assumes reinfection of previous states with the incoming strain.
            # will pull `repeat_inf_rate`% people from all previous waning compartments before the current
            for waning_compartment in waning_compartments_with_strain[::-1]:
                immune_history_dist[
                    :, multi_strain_state, 0, waning_compartment
                ] += np.sum(
                    repeat_inf_rate
                    * immune_history_dist[
                        :, prev_state, 0, waning_compartment + 1 :
                    ],  # waning_compartment + 1 selects prev waning compartments
                    axis=2,
                )
                # TODO remove 0s
                immune_history_dist[:, prev_state, 0, :] -= (
                    repeat_inf_rate
                    * immune_history_dist[
                        :, prev_state, 0, waning_compartment + 1 :
                    ]
                )
        immune_states.append(single_strain_state)
        immune_states = immune_states + multi_strain_states
    # now that we have taken all the strain stratified ages and waning compartments
    # place the fully susceptible people into [:, 0, 0].
    partial_immunity_proportion = np.sum(immune_history_dist, axis=(1, 2, 3))
    fully_susceptible_by_age = 1 - partial_immunity_proportion
    # TODO remove 0
    immune_history_dist[:, 0, 0, 0] = fully_susceptible_by_age
    return immune_history_dist


def past_immune_dist_from_serology_demographics(
    sero_path,
    age_path,
    age_limits,
    waning_times,
    num_waning_compartments,
    max_vaccine_count,
    num_strains,
    initialization_date=datetime.date(2022, 2, 12),
):
    """
    DEPRECATED: USE ABM INFORMED INITIALIZATION ROUTINES

    initializes and returns the immune history for a model based on __covid__ serological data.

    Parameters
    ----------
    sero_path: str
          relative or absolute path to serological data from which to initialize compartments
    age_path: str
          relative or absolute path to demographic data folder for age distributions
    age_limits: list(int)
          The age limits of your model that you wish to initialize compartments of.
          Example: for bins of 0-17, 18-49, 50-64, 65+ age_limits = [0, 18, 50, 65]
    waning_times: list(int)
          Time in days it takes for a person to wane from a waning compartment to the next level of protection.
          len(waning_times) == num_waning_compartments, ending in 0.
    num_waning_compartments: int
          number of waning compartments in your model that you wish to initialize.
    max_vaccination_count: int
          maximum number of vaccinations you want to actively keep track of.
          example val 2: keep track of 0, 1, 2+ shots.
    num_strains: int
          number of strains in your model that you wish to initialize.
          Note: people will be distributed across 3 strains if num_strains >= 3
          The 3 strains account for omicron, delta, and alpha waves.
          The total number of cells used to represent immune history of all strains = 2^num_strains
          if num_strains < 3, will collapse earlier strains into one another.

    Returns
    ----------
    immune_history_dist: np.array
        the proportions of the total population for each age bin stratified by immune history (natural and vaccine).
        immune history consists of previous infection history as well as number of vaccinations.
        The more recent of infection vs vaccination decides the waning compartment of that individual.
    """
    # we will need population data for weighted averages
    age_distributions = np.loadtxt(
        age_path + "United_States_country_level_age_distribution_85.csv",
        delimiter=",",
        dtype=np.float64,
        skiprows=0,
    )
    # serology data only comes in these age bins, exclusive, min age 0
    serology_age_limits = [18, 50, 65]
    (
        num_historical_strains,
        historical_time_breakpoints,
    ) = set_serology_timeline(num_strains)
    # prep the sero data into daily resolution, pass historical breakpoints to mark the strain
    # that each day of sero contributes to.
    serology = prep_serology_data(
        sero_path, num_historical_strains, historical_time_breakpoints
    )
    # age_to_diff_dict will be used to average age bins when our datas age bins collide with serology datas
    # for example hypothetical 10-20 age bin, needs to be weighted average of 0-17 and 18-49 age bins based on population
    age_to_sero_dict = {}
    age_groups = generate_yearly_age_bins_from_limits(age_limits)

    # return these after filling it with the proprtion of individuals
    # exposed to each strain of the total population
    strain_exposure_distribution = np.zeros(
        (len(age_limits), num_strains, num_waning_compartments)
    )
    # begin at the initialization date, move back from there
    prev_waning_compartment_date = initialization_date
    # for each waning index fill in its (age x strain) matrix based on weighted sero data for that age bin
    for waning_index, waning_time in zip(
        range(0, num_waning_compartments), waning_times
    ):
        # go back `waning_time` days at a time and use our diff columns to populate recoved/waning
        # initialization_date is the date our chosen serology begins, based on post-omicron peak.
        waning_compartment_date = prev_waning_compartment_date - (
            datetime.timedelta(days=waning_time)
        )
        # if the waning time for this compartment is zero, we never wane out of this compartment
        # select one day back, remember time slices are inclusive on BOTH sides!
        if waning_compartment_date == prev_waning_compartment_date:
            select = serology.loc[
                waning_compartment_date
                - datetime.timedelta(days=1) : prev_waning_compartment_date
                - datetime.timedelta(days=1)
            ]
        else:
            # grab a time range for construction of the waning compartment
            select = serology.loc[
                waning_compartment_date : prev_waning_compartment_date
                - datetime.timedelta(days=1)
            ]
        assert (
            len(select) > 0
        ), "serology data does not exist for this waning date " + str(
            waning_compartment_date
        )
        # we have now selected the information for current waning compartment, set the pointer here for next loop
        prev_waning_compartment_date = waning_compartment_date
        # `select` is now an array spaning from the beginning of the current compartment, up until the begining of the previous one.
        # however, this compartment can span multiple strains, depending on its size, do calculations for each strain!
        for strain_select in select["strain_select"].unique():
            select_strained = select[select["strain_select"] == strain_select]

            # fill our age_to_sero_dict so each age maps to its sero change we just selected
            # if we are in the last waning compartment, use sero-prevalence at that date instead
            # effectively combining all persons with previous infection on or before that date together
            for age in range(85):
                if age < serology_age_limits[0]:
                    age_to_sero_dict[age] = (
                        sum(select_strained["0_17_diff"])
                        if waning_index < num_waning_compartments - 1
                        else max(
                            select_strained[
                                "Rate (%) [Anti-N, 0-17 Years Prevalence]"
                            ]
                        )
                    )
                elif age < serology_age_limits[1]:
                    age_to_sero_dict[age] = (
                        sum(select_strained["18_49_diff"])
                        if waning_index < num_waning_compartments - 1
                        else max(
                            select[
                                "Rate (%) [Anti-N, 18-49 Years Prevalence, Rounds 1-30 only]"
                            ]
                        )
                    )
                elif age < serology_age_limits[2]:
                    age_to_sero_dict[age] = (
                        sum(select_strained["50_64_diff"])
                        if waning_index < num_waning_compartments - 1
                        else max(
                            select[
                                "Rate (%) [Anti-N, 50-64 Years Prevalence, Rounds 1-30 only]"
                            ]
                        )
                    )
                else:
                    age_to_sero_dict[age] = (
                        sum(select_strained["65_diff"])
                        if waning_index < num_waning_compartments - 1
                        else max(
                            select[
                                "Rate (%) [Anti-N, 65+ Years Prevalence, Rounds 1-30 only]"
                            ]
                        )
                    )
            # finally, sum over age groups, weighting sero by the population of each age.
            for age_group_idx, age_group in enumerate(age_groups):
                serology_age_group = [
                    age_to_sero_dict[age] for age in age_group
                ]
                population_age_group = [
                    age_distributions[age][1] for age in age_group
                ]
                serology_weighted = np.average(
                    serology_age_group, weights=population_age_group
                )
                # add to a waning compartment
                strain_exposure_distribution[
                    age_group_idx, strain_select, waning_index
                ] = serology_weighted
    # we now have the timing of when each proportion of the population was exposed to each strain
    # lets make some assumptions about repeat infections to produce immune history.
    immune_history_dist = imply_immune_history_dist_from_strains(
        strain_exposure_distribution, num_strains, num_historical_strains
    )
    # TODO add vaccinations here too.

    return immune_history_dist


def past_immune_dist_from_abm(
    abm_path,
    num_age_groups,
    age_limits,
    max_vax_count,
    waning_times,
    num_waning_compartments,
    num_strains,
    STRAIN_IDXs,
):
    """
    A function used to initialize susceptible and partially susceptible distributions for a model via ABM (agent based model) data.
    Given a path to an ABM state as CSV (schema for this data specified in README), read in dataframe, bin individuals according
    to model parameters (age/wane/vax binning), and place individuals into strata.
    Finally normalize by age group such that proportions within a single bin sum to 1.

    Parameters
    ----------
    abm_path: str
        path to the abm input data, stored as a csv.
    num_age_groups: int
        number of age bins in the model being initialized.
    age_limits: list(int)
        The age limits of your model that you wish to initialize compartments of.
        Example: for bins of 0-17, 18-49, 50-64, 65+ age_limits = [0, 18, 50, 65]
    max_vax_count: int
        the number of doses maximum before all subsequent doses are no longer counted. ex: 2 -> 0, 1, 2+ doses (3 bins)
    waning_times: list(int)
        Time in days it takes for a person to wane from a waning compartment to the next level of protection.
        len(waning_times) == num_waning_compartments, ending in 0.
    num_waning_compartments: int
        The number of waning bins in the model being initialized.
    num_strains: int
        number of distinct strains in your model, used to inform the `state` column in output
    STRAIN_IDX: intEnum
        an enum containing the name of each strain and its associated strain index, as initialized by ConfigBase.


    Returns:
    A numpy matrix stratified by age bin, immune history, vaccine bin, and waning bin. Where proportions within an single age bin sum to 1.
    Representing the distributions of people within that age bin who belong to each strata of immune history, vaccination, and waning.
    """
    num_immune_hist = 2**num_strains
    abm_population = pd.read_csv(abm_path)
    # remove those with active infections, those are designated for exposed/infected
    abm_population = abm_population[abm_population["TSLIE"] >= 0]
    abm_population = prep_abm_data(
        abm_population,
        max_vax_count,
        age_limits,
        waning_times,
        num_strains,
        STRAIN_IDXs,
    )
    immune_hist = np.zeros(
        (
            num_age_groups,
            num_immune_hist,
            max_vax_count + 1,
            num_waning_compartments,
        )
    )
    # get the number of people who fall in each age_bin, state, vax_bin, and waning_bin combination
    stratas, counts = np.unique(
        abm_population[
            ["age_bin", "state", "vax_bin", "waning_compartment_bin"]
        ],
        axis=0,
        return_counts=True,
    )
    # place people into their correct bins using the counts from above
    for strata, count in zip(stratas, counts):
        age_bin, state, vax_bin, waning_compartment_bin = strata
        immune_hist[age_bin, state, vax_bin, waning_compartment_bin] += count

    pop_by_age_bin = np.sum(immune_hist, axis=(1, 2, 3))
    # normalize for each age bin, all individual age bins sum to 1.
    immune_hist_normalized = (
        immune_hist / pop_by_age_bin[:, np.newaxis, np.newaxis, np.newaxis]
    )
    return immune_hist_normalized


def init_infections_from_abm(
    abm_path,
    num_age_groups,
    age_limits,
    max_vax_count,
    waning_times,
    num_strains,
    STRAIN_IDXs,
):
    """
    A function that uses ABM state data to inform initial infections and distribute them across infected and exposed compartments
    according to the ratio of exposed_to_infectious time and infectious_period time.
    Returns proportions of new infections belonging to each strata, all attributed to STRAIN_IDX.omicron as that was the dominant
    strain during the initialization date.

     Parameters
    ----------
    abm_path: str
        path to the abm input data, stored as a csv.
    num_age_groups: int
        number of age bins in the model being initialized.
    age_limits: list(int)
        The age limits of your model that you wish to initialize compartments of.
        Example: for bins of 0-17, 18-49, 50-64, 65+ age_limits = [0, 18, 50, 65]
    max_vax_count: int
        the number of doses maximum before all subsequent doses are no longer counted. ex: 2 -> 0, 1, 2+ doses (3 bins)
    waning_times: list(int)
        Time in days it takes for a person to wane from a waning compartment to the next level of protection.
        len(waning_times) == num_waning_compartments, ending in 0.
    num_waning_compartments: int
        The number of waning bins in the model being initialized.
    num_strains: int
        number of distinct strains in your model, used to inform the `state` column in output
    STRAIN_IDX: intEnum
        an enum containing the name of each strain and its associated strain index, as initialized by ConfigBase.

    Returns
    ----------
    (infections, exposed, infected): tuple of np.arrays
    infections = exposed + infected
    representing the proportions of each initial infection belonging to each strata, meaning sum(infections) == 1.
    All numpy arrays stratified by age, immune history, vaccination, and infecting strain (always omicron).
    """
    num_immune_hist = 2**num_strains
    abm_population = pd.read_csv(abm_path)
    # select for those with active infections, aka TSLIE < 0
    active_infections_abm = abm_population[abm_population["TSLIE"] < 0]
    # since we are looking at active infections, the last element in the strains array will be the current infecting strain
    # thus we separate it into its own column so it does not soil the immune history pre-infection of the individual
    active_infections_abm["infecting_strain"] = active_infections_abm[
        "strains"
    ].apply(lambda x: convert_strain(x.split(",")[-1], STRAIN_IDXs))

    active_infections_abm["strains"] = active_infections_abm["strains"].apply(
        lambda x: ",".join(x.split(",")[:-1])
    )
    active_infections_abm = prep_abm_data(
        active_infections_abm,
        max_vax_count,
        age_limits,
        waning_times,
        num_strains,
        STRAIN_IDXs,
    )
    infections = np.zeros(
        (
            num_age_groups,
            num_immune_hist,
            max_vax_count + 1,
            num_strains,
        )
    )
    stratas, counts = np.unique(
        active_infections_abm[
            ["age_bin", "state", "vax_bin", "infecting_strain"]
        ],
        axis=0,
        return_counts=True,
    )
    for strata, count in zip(stratas, counts):
        age_bin, state, vax_bin, infecting_strain = strata
        infections[age_bin, state, vax_bin, infecting_strain] += count

    total_pop = np.sum(infections, axis=(0, 1, 2, 3))
    # normalize so all infections sum to 1, getting proportions of each strata
    infections_normalized = infections / total_pop
    # column called "infectious" == 1 if person is actively infectious, 0 if just exposed and not yet infectious
    infected_to_exposed_ratio = sum(active_infections_abm["infectious"]) / len(
        active_infections_abm
    )
    exposed = infections_normalized * (1 - infected_to_exposed_ratio)
    infected = infections_normalized * infected_to_exposed_ratio

    return infections_normalized, exposed, infected


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# CONTACT MATRIX CODE
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


def rho(M: np.ndarray) -> np.ndarray:
    return np.max(np.real(np.linalg.eigvals(M)))


def make_two_settings_matrices(
    path_to_population_data: str,
    path_to_settings_data: str,
    region: str = "United States",
) -> tuple[np.ndarray, pd.DataFrame]:
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
            pop_prop_slice = pop_proportions[grp_out] / np.sum(
                pop_proportions[grp_out]
            )
            pop_prop_slice = np.reshape(pop_prop_slice.to_numpy(), (-1, 1))
            grouped_CM[i, j] = np.sum(pop_prop_slice * cm_slice)
    # Population proportions in each age group
    N_age = [np.sum(pop_proportions[group]) for group in age_groups]
    return (grouped_CM, N_age)


def load_demographic_data(
    demographics_path,
    regions,
    num_age_groups,
    minimum_age,
    age_limits,
) -> dict[str, dict[str, list[np.ndarray, np.float64, list[float]]]]:
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
    print("Loading demography data...")
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
