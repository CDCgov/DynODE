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

import numpy as np
import pandas as pd

# these are the configs that will be copied into each state-level directory
# their REGIONS key will be modified to match the state they work with.
# CONFIG_MOLDS = [
#     "docker_template_configs/config_global.json",
#     "docker_template_configs/config_inferer_covid.json",
#     "docker_template_configs/config_initializer_covid.json",
#     "docker_template_configs/config_runner_covid.json",
#     "docker_template_configs/config_interpreter_covid.json",
# ]
CONFIG_MOLDS = [
    "exp/fifty_state_2304_2404_3strain/config_global_template.json",
    "exp/fifty_state_2304_2404_3strain/config_inferer_covid_template.json",
    "exp/fifty_state_2304_2404_3strain/config_initializer_sero_template.json",
    "exp/fifty_state_2304_2404_3strain/config_interpreter_covid_template.json",
]
PREVIOUS_EPOCHS = "/output/fifty_state_2202_2307_3strain/smh_epoch_1_240516_2/"


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

    # Create subdirectories for each state
    for state in state_names:
        state_dir = os.path.join(dir, state)
        if not os.path.exists(state_dir):
            os.makedirs(state_dir)
        state_output_dir = os.path.join(state_dir, "output")
        if not os.path.exists(state_output_dir):
            os.makedirs(state_output_dir)


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
                # for this experiment we base our priors on the previous epochs posteriors
                # thus we modify the inferer parameters only
                if "infer" in config_file_path:
                    state_config = populate_parameters_from_previous_epoch(
                        state_config, subdir
                    )

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


def populate_parameters_from_previous_epoch(state_config, state_usps):
    previous_epoch_path = os.path.join(
        PREVIOUS_EPOCHS, "%s/checkpoint.json" % state_usps
    )
    prev_epoch_json = json.load(open(previous_epoch_path, "r"))
    # first lets modify the xbb distribution via the posteriors
    posterior_xbb_r0 = np.array(prev_epoch_json["STRAIN_R0s_2"]).flatten()
    xbb_beta_a, xbb_beta_b = fit_new_beta((posterior_xbb_r0 - 2) / 2)
    current_xbb_prior = state_config["STRAIN_R0s"][0]["params"][
        "base_distribution"
    ]
    current_xbb_prior["params"]["concentration0"] = xbb_beta_b
    current_xbb_prior["params"]["concentration1"] = xbb_beta_a
    # second we modify the MIN_HOMOLOGOUS_IMMUNITY via the posteriors
    posterior_homologous = np.array(
        prev_epoch_json["MIN_HOMOLOGOUS_IMMUNITY"]
    ).flatten()
    homologous_beta_a, homologous_beta_b = fit_new_beta(posterior_homologous)
    current_homologous_prior = state_config["MIN_HOMOLOGOUS_IMMUNITY"][
        "params"
    ]
    current_homologous_prior["concentration0"] = homologous_beta_b * 1.3
    current_homologous_prior["concentration1"] = homologous_beta_a * 1.3
    # thirdly we modify all the seasonality parameters based on the posteriors
    ## amplitude
    posterior_amp = np.array(
        prev_epoch_json["SEASONALITY_AMPLITUDE"]
    ).flatten()
    amp_beta_a, amp_beta_b = fit_new_beta((posterior_amp - 0.02) / 0.18)
    current_amp_prior = state_config["SEASONALITY_AMPLITUDE"]["params"][
        "base_distribution"
    ]["params"]
    current_amp_prior["concentration0"] = amp_beta_b
    current_amp_prior["concentration1"] = amp_beta_a
    ## shift (this is trunc normal)
    posterior_shift = np.array(prev_epoch_json["SEASONALITY_SHIFT"]).flatten()
    shift_mean, shift_sd = np.average(posterior_shift), np.std(posterior_shift)
    current_shift_prior = state_config["SEASONALITY_SHIFT"]["params"]
    current_shift_prior["loc"] = shift_mean
    current_shift_prior["scale"] = shift_sd
    ## second wave
    # posterior_second_wave = np.array(
    #     prev_epoch_json["SEASONALITY_SECOND_WAVE"]
    # ).flatten()
    # second_wave_beta_a, second_wave_beta_b = fit_new_beta(
    #     posterior_second_wave
    # )
    # current_second_wave_prior = state_config["SEASONALITY_SECOND_WAVE"][
    #     "params"
    # ]
    # current_second_wave_prior["concentration0"] = second_wave_beta_b
    # current_second_wave_prior["concentration1"] = second_wave_beta_a
    # ihrs
    for i in range(4):
        ihr_i_posteriors = np.array(prev_epoch_json["ihr_%s" % i]).flatten()
        ihr_i_beta_a, ihr_i_beta_b = fit_new_beta(ihr_i_posteriors)
        state_config["ihr_%s" % i] = {
            "distribution": "Beta",
            "params": {
                "concentration1": ihr_i_beta_a,
                "concentration0": ihr_i_beta_b,
            },
        }

    # immune mult
    ihr_mult_posteriors = np.array(
        prev_epoch_json["ihr_immune_mult"]
    ).flatten()
    ihr_mult_beta_a, ihr_mult_beta_b = fit_new_beta(ihr_mult_posteriors)
    state_config["ihr_immune_mult"] = {
        "distribution": "Beta",
        "params": {
            "concentration1": ihr_mult_beta_a,
            "concentration0": ihr_mult_beta_b,
        },
    }

    return state_config


def fit_new_beta(posterior_samples):
    m = np.average(posterior_samples)
    v = np.var(posterior_samples)

    a = (m * (1 - m) / v - 1) * m
    b = (m * (1 - m) / v - 1) * (1 - m)
    return a, b


def get_all_codes():
    return list(state_names["stusps"])


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
    pops.loc[-1] = [
        "US",
        "United States",
        sum(pops["POPULATION"]),
        None,
        None,
    ]
    args = parser.parse_args()
    exp = args.exp
    states = args.states
    if "all" in states:
        states = get_all_codes()
        states.remove("US")
        states.remove("DC")
    config_molds = args.config_molds
    create_state_subdirectories(exp, states)
    populate_config_files(exp, config_molds)
    print(
        "Created and populated state level directories with read-only copies of the config files"
    )
