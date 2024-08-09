"""
A basic script that sets up experiments by taking a directory, creating a bunch of
state-specific folders within it and populating each folder with read-only copies
of the configuration files specified. This way each state can be run in parallel
where a single state only needs its config files and can store output within its
state-specific folder, to be collected later.
"""

import argparse
import json
import os

import pandas as pd

# these are the configs that will be copied into each state-level directory
# their REGIONS key will be modified to match the state they work with.
CONFIG_MOLDS = [
    "exp/example_azure_experiment/example_template_configs/config_global.json",
    "exp/example_azure_experiment/example_template_configs/config_inferer_covid.json",
    "exp/example_azure_experiment/example_template_configs/config_initializer_covid.json",
    "exp/example_azure_experiment/example_template_configs/config_runner_covid.json",
    "exp/example_azure_experiment/example_template_configs/config_interpreter_covid.json",
]
EXPERIMENT_DIRECTORY = "exp"


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

                # Create a new read-only copy of the JSON file with modified data
                new_config_file_path = os.path.join(
                    subdir_path, os.path.basename(config_file_path)
                )
                # if the config file already exists, we remove and override it.
                if os.path.exists(new_config_file_path):
                    # change back from readonly so it can be deleted, otherwise get PermissionError
                    os.chmod(new_config_file_path, 0o777)
                    os.remove(new_config_file_path)

                with open(new_config_file_path, "w") as f:
                    json.dump(state_config, f, indent=4)

                # Set the new file permissions to read-only
                os.chmod(new_config_file_path, 0o444)


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
# experiment directory
parser.add_argument(
    "-e",
    "--experiment_name",
    type=str,
    required=True,
    help="str, name of the experiment states should be placed into",
)
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
    experiment_name = args.experiment_name
    exp = os.path.join(EXPERIMENT_DIRECTORY, experiment_name)
    if "all" in states:
        states = get_all_codes()
    config_molds = args.config_molds
    create_state_subdirectories(exp, states)
    populate_config_files(exp, config_molds)
    print(
        "Created and populated state level directories with read-only copies of the config files"
    )
