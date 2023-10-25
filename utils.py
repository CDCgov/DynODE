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
def sample_r0(R0_means):
    """sample r0 for each strain according to an exponential distribution
    with rate equal to inverse of strain specific r0 in config file

    Parameters
    ----------
    R0_means: list(int)
        list of mean R0 for each pathogen strain, from which exponential distribution will center around.
        len(R0_means) = # of strains in your model."""
    r0s = []
    for i, sample_mean in enumerate(R0_means):
        excess_r0 = numpyro.sample(
            "excess_r0_" + str(i), dist.Exponential(1 / sample_mean)
        )
        r0 = numpyro.deterministic("r0_" + str(i), 1 + excess_r0)
        r0s.append(r0)
    return r0s


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
# DEMOGRAPHICS CODE
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


def generate_yearly_age_bins_from_limits(age_limits):
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
    Does this by searching for age demographics data in path."""
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
                age_bin_pop += age_distributions[current_age][
                    1
                ]  # get the population
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


def prep_serology_data(path, waning_time):
    """
    reads serology data from path, filters to only USA site,
    filters Date Ranges from Sep 2020 - Feb 2022,
    calculates monotonically increasing rates of change (to combat sero-reversion from the assay),
    and converts string dates to datetime.dt.date objects

    TODO: change method of combatting sero-reversion to one outlined here:
    https://www.nature.com/articles/s41467-023-37944-5

    Parameters
    ----------
    waning_protect_means: str
        relative path to serology data sourced from
        https://data.cdc.gov/Laboratory-Surveillance/Nationwide-Commercial-Laboratory-Seroprevalence-Su/d2tw-32xv

    Returns
    ----------
    serology table containing the following additional columns:
    `collection_start` = assay collection start date
    `collection_end` = assay collection end date
    `age0_age1_diff` = difference in `Rate (%) [Anti-N, age1-age2 Years Prevalence]` from current and previous collection.
                  enforced to be positive or 0 to combat sero-reversion. Columns repeats for age bins [0-17, 18-49, 50-64, 65+]
    modifies `Rate (%) [Anti-N, age1-age2 Years Prevalence, Rounds 1-30 only]` columns to enforce monotonicity as well.
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
    # TODO possible reimplementation of variable waning compartment bin width
    # will probably need to return to [::-x] slicing with a variable x or something.
    serology = (
        serology.resample("1d")  # downsample to daily freq
        .interpolate()  # linear interpolate between days
        .resample(
            str(waning_time) + "d", origin="end"
        )  # resample to waning compart width
        .max()
    )
    # we will use the absolute change in % serology prevalence to initalize wane compartments
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


def load_serology_demographics(
    sero_path,
    age_path,
    age_limits,
    waning_time,
    num_waning_compartments,
    num_strains,
):
    """
    initalizes and returns the recovered and waning compartments for a model based on serological data.

    Parameters
    ----------
    sero_path: str
          relative or absolute path to serological data from which to initalize compartments
    age_path: str
          relateive or absolute path to demographic data folder for age distributions
    age_limits: list(int)
          The age limits of your model that you wish to initalize compartments of.
          Example: for bins of 0-17, 18-49, 50-64, 65+ age_limits = [0, 18, 50, 65]
    waning_time: int
          Time in days it takes for a person to wane to the next level of protection
    num_waning_compartments:
          number of waning compartments in your model that you wish to initalize.
    num_strains:
          number of strains in your model that you wish to initalize.
          Note: people will be distributed across 3 strains if num_strains >= 3
          The 3 strains account for omicron, delta, and alpha waves. And are timed accordingly.
          if num_strains < 3, will collapse earlier strains into one another.
    """
    # TODO we now have daily precision via linear interpolation,
    # implement waning compartments with two strains filled in
    initalization_date = datetime.date(2022, 2, 11)
    serology = prep_serology_data(sero_path, waning_time)
    # serology = serology.resample(str(waning_time) + "d")
    # we will need population data for weighted averages
    age_distributions = np.loadtxt(
        age_path + "United_States_country_level_age_distribution_85.csv",
        delimiter=",",
        dtype=np.float64,
        skiprows=0,
    )
    # serology data only comes in these age bins, inclusive
    serology_age_limits = [17, 49, 64]
    # number of strains alloted for in the serological data, for covid this is omicron, delta, and alpha
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
    ), "set breakpoints for each of the historical strains you want to initalize with sero data"
    # age_to_diff_dict will be used to average age bins when our datas age bins collide with serology datas
    # for example 0-4 age bin will fit inside 0-17,
    # but what about hypothetical 10-20 age bin? This needs to be weighted average of 0-17 and 18-49 age bins
    age_to_sero_dict = {}
    age_groups = generate_yearly_age_bins_from_limits(age_limits)

    # return these after filling it with the %s of waned/recovered individuals
    waning_init_distribution = np.zeros(
        (len(age_limits), num_strains, num_waning_compartments)
    )
    recovered_init_distribution = np.zeros(
        shape=(len(age_limits), num_strains)
    )
    # for each waning index fill in its age x strain matrix based on weighted sero data for that age bin
    for waning_index in range(0, num_waning_compartments + 1):
        # now we go back `waning_time` days at a time and use our diff columns to populate recoved/waning
        # initalization_date is the date our chosen serology begins, based on post-omicron peak.
        waning_compartment_date = initalization_date - (
            datetime.timedelta(days=waning_time) * (waning_index)
        )
        select = serology.loc[waning_compartment_date:waning_compartment_date]
        assert (
            len(select) > 0
        ), "serology data does not exist for this waning date " + str(
            waning_compartment_date
        )
        # depending how far back we are looking, we may be filling waning information for past strains
        # omicron = strain 2, delta = 1, alpha = 0 for example
        strain_select = (
            num_historical_strains - 1
        )  # initalize as most recent strain
        for historical_breakpoint in historical_time_breakpoints:
            # TODO check this make sure we arent off by one here with <= vs <
            if waning_compartment_date <= historical_breakpoint:
                strain_select -= 1
        # select the only row as a series
        select = select.iloc[0]
        # fill our age_to_sero_dict so each age maps to its sero change we just selected
        for age in range(85):
            if age <= serology_age_limits[0]:
                age_to_sero_dict[age] = (
                    select["0_17_diff"]
                    if waning_index < num_waning_compartments
                    else select["Rate (%) [Anti-N, 0-17 Years Prevalence]"]
                )
            elif age <= serology_age_limits[1]:
                age_to_sero_dict[age] = (
                    select["18_49_diff"]
                    if waning_index < num_waning_compartments
                    else select[
                        "Rate (%) [Anti-N, 18-49 Years Prevalence, Rounds 1-30 only]"
                    ]
                )
            elif age <= serology_age_limits[2]:
                age_to_sero_dict[age] = (
                    select["50_64_diff"]
                    if waning_index < num_waning_compartments
                    else select[
                        "Rate (%) [Anti-N, 50-64 Years Prevalence, Rounds 1-30 only]"
                    ]
                )
            else:
                age_to_sero_dict[age] = (
                    select["65_diff"]
                    if waning_index < num_waning_compartments
                    else select[
                        "Rate (%) [Anti-N, 65+ Years Prevalence, Rounds 1-30 only]"
                    ]
                )
        for age_group_idx, age_group in enumerate(age_groups):
            serology_age_group = [age_to_sero_dict[age] for age in age_group]
            population_age_group = [
                age_distributions[age][1] for age in age_group
            ]
            serology_weighted = np.average(
                serology_age_group, weights=population_age_group
            )
            # this is where we would uniformly spread out waning if we wanted to
            if (
                waning_index == 0
            ):  # waning_index=0 -> add to recovered compartment
                recovered_init_distribution[
                    age_group_idx, strain_select
                ] = serology_weighted
            else:  # add to a waning compartment, subtract 1 to fill 0th waning compartment
                waning_init_distribution[
                    age_group_idx, strain_select, waning_index - 1
                ] = serology_weighted

    return recovered_init_distribution, waning_init_distribution


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
        type(minimum_age), int
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
    Loads demography data for the specified FIPS regions

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
            sch_CM_all, oth_CM_all, region_data = make_two_settings_matrices(
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
            # Create the age-grouped other setting contact
            oth_CM, N_age_oth = create_age_grouped_CM(
                region_data,
                oth_CM_all,
                num_age_groups,
                minimum_age,
                age_limits,
            )
            # Save one of the two N_ages (they are the same) in a new N_age var
            N_age = N_age_sch
            # Rescale contact matrices by leading eigenvalue
            oth_CM = oth_CM / rho(oth_CM)
            sch_CM = sch_CM / rho(sch_CM)
            # Transform Other cm with the new age limits [NB: to transpose?]
            region_demographic_data_dict = {
                "sch_CM": sch_CM.T,
                "oth_CM": oth_CM.T,
                "N_age": np.array(N_age),
                "N": np.array(np.sum(N_age)),
            }
            demographic_data[r] = region_demographic_data_dict
        except Exception as e:
            print(
                f"Something went wrong with {region} and produced the following error:\n\t{e}"
            )
    return demographic_data
