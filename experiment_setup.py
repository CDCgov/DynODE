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
    "config/config_global.json",
    "config/config_inferer_covid.json",
    "config/config_initializer_covid.json",
    "config/config_runner_covid.json",
]


def create_state_subdirectories(dir, states_fips):
    """
    function to create an experiment directory `dir` and then create
    subfolders for each FIPS code in `states`.
    Will not override if `dir` or `dir/state_fips[i]` already exists

    Parameters
    ------------
    `dir`: str
        relative or absolute directory path of the experiment,
        for which subdirectories per state will be created under it.

    `state_fips`: list[int]
        list of fips codes per state involved in the experiment, will create subfolders of `dir`
        with each fips code.

    Returns
    ------------
    None
    """
    # Create the main directory if it does not exist
    if not os.path.exists(dir):
        os.makedirs(dir)

    # Create subdirectories for each state
    for state in states_fips:
        state_dir = os.path.join(dir, str(state))
        if not os.path.exists(state_dir):
            os.makedirs(state_dir)


def populate_config_files(dir, configs):
    """
    scans an experiment directory `dir` opening each folder, and copying over read-only versions
    of each json file in `configs`, modifying the "REGIONS" key to mirror the state to which the FIPS code is assigned.

    will raise an error if a subdirectory of `dir` is not a fips code able to be parsed to an integer.

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
    for subdir in os.listdir(dir):
        subdir_path = os.path.join(dir, subdir)
        if os.path.isdir(subdir_path):
            state_name = fips_to_state(int(subdir))

            for config_file_path in configs:
                # Read the original JSON file
                with open(config_file_path) as f:
                    data = json.load(f)

                # Change the "REGION" key to state name
                data["REGIONS"] = [state_name]

                # Create a new read-only copy of the JSON file with modified data
                new_config_file_path = os.path.join(
                    subdir_path, os.path.basename(config_file_path)
                )
                # if the config file already exists, we remove and override it.
                if os.path.exists(new_config_file_path):
                    os.remove(new_config_file_path)

                with open(new_config_file_path, "w") as f:
                    json.dump(data, f, indent=4)

                # Set the new file permissions to read-only
                os.chmod(new_config_file_path, 0o444)


def fips_to_state(fips):
    """
    basic function to read in an integer fips code and return associated state name

    Parameters
    ----------
    fips: int
        fips code the state

    Returns
    ----------
    str/KeyError: state name, or KeyError if fips does not point to a state or isnt an integer
    """
    state_info = states[states["st"] == fips]
    if len(state_info) == 1:
        return state_info["stname"].iloc[0]
    else:
        raise KeyError("Unknown fips %d" % fips)


# script takes arguments to specify the experiment being created.
parser = argparse.ArgumentParser()
# experiment directory
parser.add_argument(
    "-e",
    "--exp",
    type=str,
    required=True,
    help="str, directory experiment should be placed in, relative to this file",
)
# list of fips codes
parser.add_argument(
    "-f",
    "--fips",
    type=int,
    required=True,
    nargs="+",
    help="space separated list of integers representing FIPS code of each state",
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
    states = pd.read_csv("data/fips_to_name.csv")
    args = parser.parse_args()
    exp = args.exp
    fips = args.fips
    config_molds = args.config_molds
    create_state_subdirectories(exp, fips)
    populate_config_files(exp, config_molds)
    print(
        "Created and populated state level directories with read-only copies of the config files"
    )
