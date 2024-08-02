"""
A basic script that sets up experiments by taking a directory, creating a bunch of
state-specific folders within it and populating each folder with read-only copies
of the configuration files specified. This way each state can be run in parallel
where a single state only needs its config files and can store output within its
state-specific folder, to be collected later.
"""

import argparse
import copy
import json
import os

import numpy as np
import pandas as pd

CONFIG_MOLDS = [
    "exp/projections_2407_2507/template_configs/config_global.json",
    "exp/projections_2407_2507/template_configs/scenario_template_wseason.json",
]

EXP_FOLDER = "exp/projections_2407_2507"
SCEN_CSV = "exp/projections_2407_2507/scenarios.csv"


def create_state_subdirectories(dir, state_names):
    """
    function to create an experiment directory `dir` and then create
    subfolders for each Postal Abreviation in `state_names`.
    Will not override if `dir` or `dir/state_names[i]` already exists

    Parameters
    ------------
    `dir`: str
        relative or absolute directory path of the experiment,
        for which subdirectories per state will be created under it.

    `state_names`: list[str]
        list of USPS postal codes per state involved in the experiment, will create subfolders of `dir`
        with each code.

    Returns
    ------------
    None
    """
    # Create the main directory if it does not exist
    if not os.path.exists(dir):
        os.makedirs(dir)

    # Create subdirectories for each state inside the "states" folder
    for state in state_names:
        state_dir = os.path.join(dir, "states", state)
        if not os.path.exists(state_dir):
            os.makedirs(state_dir)


def create_multiple_scenarios_configs(state_config, state_abb, subdir_path):
    """
    Create json for scenarios based on SCEN_CSV. This would include:
    1. intro_time, when does the new strain get introduced
    2. vaccination, 0 or 1, whether this is recommended/implemented
    3. immune_escape, scalar multiplier (in %) indicating if the immune escape of the
    next major strain is the same (e.g. 1), higher (e.g., 1.2) or lower (e.g., 0.8)
    4. vaccine_efficacy, vaccine efficacy multiplier (in %), which is used to calculate VE of
    seasonal booster via 1 - (1 - VE_2dose) * (1 - vaccine efficacy)
    """
    print(state_abb)
    intro_time_lookup = {
        "aug": [32],
        "sep": [63],
        "oct": [93],
        "nov": [124],
        "dec": [154],
        "none": [],
        "sample": [1],
    }
    df = pd.read_csv(SCEN_CSV)
    # df = df[["b" in x for x in df["id"]]]
    dicts = []
    paths = []
    for index, row in df.iterrows():
        vs = row["vaccination"]
        it = row["intro_time"]
        ie = row["immune_escape"]
        ve = row["vaccine_efficacy"]

        st_config = copy.deepcopy(state_config)
        # vax0 no booster (except children), vax1 with booster across all age
        if vs == 0:
            st_config["VACCINATION_MODEL_DATA"] = (
                "/input/data/vaccination-data/2024_06_30_to_2025_06_28_vax0/"
            )
        else:
            st_config["VACCINATION_MODEL_DATA"] = (
                "/input/data/vaccination-data/2024_06_30_to_2025_06_28_vax1/"
            )
            st_config["VACCINATION_RATE_MULTIPLIER"] = vs / 100

        # intro time convert from month to day in model
        st_config["INTRODUCTION_TIMES"] = intro_time_lookup[it]
        if it == "none":
            st_config["INTRODUCTION_SCALES"] = []
            st_config["INTRODUCTION_PCTS"] = []
        st_config["SAMPLE_STRAIN_X_INTRO_TIME"] = it == "sample"
        # inject multiplier that get used in inferer_projection
        st_config["STRAIN_INTERACTIONS"][6][4] = ie / 100
        st_config["STRAIN_INTERACTIONS"][6][5] = ie / 100
        # VE is calculated based on VE_2dose
        vaccine_efficacy = np.array(st_config["VACCINE_EFF_MATRIX"])
        vaccine_efficacy[:, 3] = 1 - (1 - vaccine_efficacy[:, 2]) * (
            1 - ve / 100
        )
        st_config["VACCINE_EFF_MATRIX"] = vaccine_efficacy.tolist()
        dicts.append(st_config)
        new_config_file_path = os.path.join(
            subdir_path, f"vs{str(vs)}_it{it}_ie{str(ie)}_ve{str(ve)}.json"
        )
        paths.append(new_config_file_path)

    return dicts, paths


def populate_config_files(dir, configs):
    """
    scans an experiment directory `dir` opening each folder, and copying over read-only versions
    of each json file in `configs`, modifying the "REGIONS" key to match the postal code.
    Modifies the `POP_SIZE` variable to match the states population according to the census.
    Modifies the `INITIAL_INFECTIONS` variable to equal the same % of the population as in the mold config.
        eg: 2% of TOTAL_POP in mold config applied to each state's individual `POP_SIZE`

    will raise an error if a subdirectory of `dir` is not a postal code able to be looked up.

    Parameters
    ------------
    `dir`: str
        relative or absolute directory path of the experiment,
        contains subdirectories created by `create_state_subdirectories`

    `configs`: list[str]
        list of paths to each config mold, these config molds will be copied into each subdirectory as read-only
        they will have their "REGIONS" key changed to resemble the state the subdirectory is modeling.

    Returns
    ------------
    None
    """
    dir = os.path.join(dir, "states")
    for subdir in os.listdir(dir):
        subdir_path = os.path.join(dir, subdir)
        json_dicts = []
        output_paths = []
        if os.path.isdir(subdir_path):
            state_name = code_to_state(subdir)
            state_pop = code_to_pop(state_name)

            for config_file_path in configs:
                # Read the original JSON file
                with open(config_file_path) as f:
                    state_config = json.load(f)

                # Change the "REGION" key to state name
                state_config["REGIONS"] = [state_name]

                if "POP_SIZE" in state_config.keys():
                    # havent changed yet, so old value still in `state_config`
                    mold_pop_size = state_config["POP_SIZE"]
                    # match the same % of the population as in the mold config to new state POP_SIZE
                    if "INITIAL_INFECTIONS" in state_config.keys():
                        mold_initial_inf = state_config["INITIAL_INFECTIONS"]
                        # state_pop * (% of infections in the mold config)
                        # round to 3 sig figs, convert to int
                        state_config["INITIAL_INFECTIONS"] = int(
                            float(
                                "%.3g"
                                % (
                                    state_pop
                                    * (mold_initial_inf / mold_pop_size)
                                )
                            )
                        )
                    # round pop sizes 3 sig figs then convert to int
                    state_config["POP_SIZE"] = int(float("%.3g" % state_pop))

                if "scenario" in config_file_path:
                    dicts, paths = create_multiple_scenarios_configs(
                        state_config, subdir, subdir_path
                    )
                    json_dicts.extend(dicts)
                    output_paths.extend(paths)
                else:
                    json_dicts.append(state_config)
                    new_config_file_path = os.path.join(
                        subdir_path, os.path.basename(config_file_path)
                    )
                    output_paths.append(new_config_file_path)

            # Create a new read-only copy of the JSON file with modified data
            for d, p in zip(json_dicts, output_paths):
                # if the config file already exists, we remove and override it.
                if os.path.exists(p):
                    # change back from readonly so it can be deleted, otherwise get PermissionError
                    os.chmod(p, 0o777)
                    os.remove(p)

                with open(p, "w") as f:
                    json.dump(d, f, indent=4)

                # Set the new file permissions to read-only
                os.chmod(p, 0o444)


def code_to_state(code):
    """
    basic function to read in an postal code and return associated state name

    Parameters
    ----------
    code: str
        usps code the state

    Returns
    ----------
    str/KeyError: state name, or KeyError if code does not point to a state or isnt an str
    """
    state_info = state_names[state_names["stusps"] == code]
    if len(state_info) == 1:
        return state_info["stname"].iloc[0]
    else:
        raise KeyError("Unknown code %s" % code)


def code_to_pop(state_name):
    """
    basic function to read in an postal code and return associated state name

    Parameters
    ----------
    state_name: str
        state name

    Returns
    ----------
    str/KeyError: state population, or KeyError if invalid state name
    """
    state_pop = pops[pops["STNAME"] == state_name]
    if len(state_pop) == 1:
        return state_pop["POPULATION"].iloc[0]
    else:
        raise KeyError("Unknown fips %s" % state_name)


def get_all_codes():
    return list(state_names["stusps"])


# script takes arguments to specify the experiment being created.
parser = argparse.ArgumentParser()

# list of fips codes
parser.add_argument(
    "-s",
    "--states",
    type=str,
    required=True,
    nargs="+",
    help="space separated list of str representing USPS postal code of each state",
)
# the molds of configs to bring into each state sub-dir
parser.add_argument(
    "-m",
    "--config_molds",
    type=str,
    required=False,
    nargs="+",
    default=CONFIG_MOLDS,
    help="space separated paths to the config molds, defaults to some in /config",
)

if __name__ == "__main__":
    state_names = pd.read_csv("data/fips_to_name.csv")
    pops = pd.read_csv("data/demographic-data/CenPop2020_Mean_ST.csv")
    # adding a USA row with the sum of all state pops
    usa_pop_row = pd.DataFrame(
        [
            [
                "US",
                "United States",
                sum(pops["POPULATION"]),
                None,
                None,
            ]
        ],
        columns=pops.columns,
    )
    pops = pd.concat([pops, usa_pop_row], ignore_index=True)
    args = parser.parse_args()
    states = args.states
    if "all" in states:
        states = get_all_codes()
        states.remove("US")
        states.remove("DC")

    config_molds = args.config_molds
    create_state_subdirectories(EXP_FOLDER, states)
    populate_config_files(EXP_FOLDER, config_molds)
    print(
        "Created and populated state level directories with read-only copies of the config files"
    )
