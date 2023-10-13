import matplotlib.pyplot as plt
import config.config_base as cf
from config.config_base import DataConfig as dc
from config.config_base import ModelConfig as mc
import pandas as pd
import numpy as np
import os, glob

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
                    np.loadtxt(elt, delimiter=",", dtype=np.float64, skiprows=0),
                )
                for elt in all_settings_data_files
                if region in elt
            ]
        )
    else:
        region_data_file = "United_States_country_level_age_distribution_85.csv"
        settings_data_dict = dict(
            [
                (
                    elt.split("_")[setting_index],
                    np.loadtxt(elt, delimiter=",", dtype=np.float64, skiprows=0),
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
        else:
            oth_CM += settings_data_dict[setting]
    return (sch_CM, oth_CM, region_data)


def create_age_grouped_CM(
    region_data: pd.DataFrame,
    setting_CM: np.ndarray,
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
    assert 0 <= dc.MINIMUM_AGE < 84, "Please correct the value of the minimum age"
    # Check if the dc.MINIMUM_AGE is an int
    assert type(dc.MINIMUM_AGE) == int, "Please make sure the minimum age is an int"
    # Check to see if the age limits specified are ordered properly
    assert (
        dc.MINIMUM_AGE < dc.AGE_LIMITS[0] < dc.AGE_LIMITS[-1] < 84
    ), "The entered age limits are not compatible"
    # Check if the upper bound of the age limits is greater than the lower bound
    assert (
        dc.AGE_LIMITS[0] < dc.AGE_LIMITS[1] + 1
    ), "The bounds for the age limits are not proper"
    # Check if there are only two age limits
    # Output and baseline age groups in the contact matrices, respectively
    n, m = dc.NUM_AGE_GROUPS, setting_CM.shape[0]
    # Create new age groups from the age limits, e.g. if [18,66], <18,18-64,65+
    age_groups = []
    for i in range(1, n + 1):
        if i == 1:
            age_groups.append(list(range(dc.MINIMUM_AGE, dc.AGE_LIMITS[0])))
        elif i == n:
            age_groups.append(list(range(dc.AGE_LIMITS[n - 2], m)))
        else:
            age_groups.append(list(range(dc.AGE_LIMITS[i - 2], dc.AGE_LIMITS[i - 1])))
    # Create the empty contact matrix for the new age groups
    grouped_CM = np.empty(
        (dc.NUM_AGE_GROUPS, dc.NUM_AGE_GROUPS), dtype=setting_CM.dtype
    )
    # Get the population data to be used for proportions in the
    pop_proportions = region_data["Population"].div(region_data["Population"].sum())
    # Fill in the age-grouped contact matrix
    for i, grp_out in enumerate(age_groups):
        for j, grp_in in enumerate(age_groups):
            cm_slice = setting_CM[np.ix_(grp_out, grp_in)]
            pop_prop_slice = pop_proportions[grp_out] / np.sum(pop_proportions[grp_out])
            pop_prop_slice = np.reshape(pop_prop_slice.to_numpy(), (-1, 1))
            grouped_CM[i, j] = np.sum(pop_prop_slice * cm_slice)
    # Population proportions in each age group
    N_age = [np.sum(pop_proportions[group]) for group in age_groups]
    return (grouped_CM, N_age)


def load_demographic_data() -> (
    dict[str, dict[str, list[np.ndarray, np.float64, list[float]]]]
):
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
    path_to_settings_data = dc.MIXING_PATH + "contact_matrices"
    path_to_population_data = dc.MIXING_PATH + "population_rescaled_age_distributions"
    # Check if the paths to the files exists
    assert os.path.exists(
        dc.MIXING_PATH
    ), f"The base path {dc.MIXING_PATH} does not exist as it should"
    assert os.path.exists(
        path_to_settings_data
    ), f"The path to the contact matrices does not exist as it should"
    assert os.path.exists(
        path_to_population_data
    ), f"The path to population-rescaled age distributions does not exist as it should"
    # Create an empty dictionary for the demographic data
    demographic_data = dict([(r, "") for r in cf.REGIONS])
    # Create contact matrices
    for r in cf.REGIONS:
        print(f"\t...for {r}")
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
            )
            # Create the age-grouped other setting contact
            oth_CM, N_age_oth = create_age_grouped_CM(
                region_data,
                oth_CM_all,
            )
            # Save one of the two N_ages (they are the same) in a new N_age var
            N_age = N_age_sch
            # Rescale contact matrices by leading eigenvalue
            oth_CM = oth_CM / rho(oth_CM)
            sch_CM = sch_CM / rho(oth_CM)
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


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# PLOTTING CODE
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


def plot_ode_solution(
    sol, plot_compartments=["s", "i", "r"], sum_across_axis=2, save_path=None
):
    sol = np.array(sol)
    get_indexes = [
        mc.idx.__getitem__(compartment.strip().upper())
        for compartment in plot_compartments
    ]
    if sum_across_axis:
        sol = sol.sum(axis=sum_across_axis)  # used to sum across age groups

    fig, ax = plt.subplots(1)
    for compartment, idx in zip(plot_compartments, get_indexes):
        ax.plot(sol[idx], label=compartment)
    fig.legend()
    if save_path:
        fig.savefig(save_path)
    return fig, ax


def plot_diffrax_solution(sol, plot_compartments=["s", "e", "i", "r"], save_path=None):
    get_indexes = []
    for compartment in plot_compartments:
        if "W" in compartment.upper():
            # waning compartments are held in a different manner, we need two indexes to access them
            index_slice = [
                mc.idx.__getitem__("W"),
                mc.w_idx.__getitem__(compartment.strip().upper()),
            ]
            get_indexes.append(index_slice)
        else:
            get_indexes.append(mc.idx.__getitem__(compartment.strip().upper()))

    fig, ax = plt.subplots(1)
    for compartment, idx in zip(plot_compartments, get_indexes):
        if "W" in compartment.upper():
            # if we are plotting a waning compartment, we need to parse 1 extra dimension
            sol_compartment = np.array(sol[idx[0]])[:, :, :, idx[1]]
        else:
            # non-waning compartments dont have this extra dimension
            sol_compartment = sol[idx]
        dimensions_to_sum_over = tuple(range(1, sol_compartment.ndim))
        ax.plot(sol_compartment.sum(axis=dimensions_to_sum_over), label=compartment)
    fig.legend()
    if save_path:
        fig.savefig(save_path)
    return fig, ax
