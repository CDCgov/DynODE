# ruff: noqa: E402
import argparse
import os
import shutil
import sys

import numpy as np

# adding things to path since in a docker container pathing gets changed
sys.path.append("/app")
sys.path.append("/app/mechanistic_model/")
# sys.path.append(".")
# sys.path.append(os.getcwd())
import jax.numpy as jnp
import pandas as pd
from epoch_two_initializer import smh_initializer_epoch_two
from inferer_smh import SMHInferer
from utils import combine_strains, combined_strains_mapping

from mechanistic_model.mechanistic_runner import MechanisticRunner
from model_odes.seip_model import seip_ode


def rework_initial_state(initial_state):
    """
    Take the original `initial_state` which is (4, 16, 3, 5), collapsing the
    strain 0 and 1 into 0, and collapsing wane 3 and 4 into 3, resulting in
    a 3-strain initial_state with (4, 8, 3, 4).
    """
    hist_map, strain_map = combined_strains_mapping(1, 0, 4)
    s_new_1 = combine_strains(
        initial_state[0], hist_map, strain_map, 4, strain_axis=False
    )[:, :8, :, :]
    s_new_2 = jnp.ones((4, 8, 3, 4)) * s_new_1[:, :, :, :4]
    s_new = s_new_2.at[:, :, :, 3].add(s_new_1[:, :, :, 4])
    e_new = combine_strains(
        initial_state[1], hist_map, strain_map, 4, strain_axis=True
    )[:, :, :, :3]
    i_new = combine_strains(
        initial_state[2], hist_map, strain_map, 4, strain_axis=True
    )[:, :, :, :3]
    c_new = initial_state[3][:, :8, :, :3]
    initial_state = (s_new, e_new, i_new, c_new)
    return initial_state


parser = argparse.ArgumentParser()
parser.add_argument(
    "-s",
    "--state",
    type=str,
    help="directory for the state to run, resembles USPS code of the state",
)

parser.add_argument(
    "-j", "--jobid", type=str, help="job-id of the state being run on Azure"
)

if __name__ == "__main__":
    args = parser.parse_args()
    jobid = args.jobid
    state = args.state
    # we are going to be rerouting stdout and stderror to files in our output blob
    stdout = sys.stdout
    stderror = sys.stderr
    save_path = "/output/fifty_state_2304_2404_3strain/%s/%s/" % (jobid, state)
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    stdout_file = open(save_path + "stdout.txt", "w")
    stderror_file = open(save_path + "stderror.txt", "w")
    try:
        sys.stdout = stdout_file
        sys.stderr = stderror_file
        model_day = 392
        # step 1: define your paths
        state_config_path = (
            "/app/exp/fifty_state_2304_2404_3strain/" + state + "/"
        )
        # state_config_path = "exp/fifty_state_sero_second_try/" + args.state + "/"
        print("Running the following state: " + state + "\n")
        # global_config include definitions such as age bin bounds and strain definitions
        # Any value or data structure that needs context to be interpretted is here.
        GLOBAL_CONFIG_PATH = state_config_path + "config_global_template.json"
        # a temporary global config that matches with original initializer
        # TODO: rework initializer to get rid of this step

        # defines the init conditions of the scenario: pop size, initial infections etc.
        INITIALIZER_CONFIG_PATH = (
            state_config_path + "config_initializer_sero_template.json"
        )
        # defines prior __distributions__ for inferring runner variables.
        INFERER_CONFIG_PATH = (
            state_config_path + "config_inferer_covid_template.json"
        )
        # defines how the solution should be viewed, what slices examined, how to save.
        INTERPRETER_CONFIG_PATH = (
            state_config_path + "config_interpreter_covid_template.json"
        )
        # save copies of the used config files to output for reproducibility purposes
        shutil.copy(GLOBAL_CONFIG_PATH, save_path + "config_global_used.json")
        shutil.copy(
            INFERER_CONFIG_PATH, save_path + "config_inferer_used.json"
        )
        shutil.copy(
            INITIALIZER_CONFIG_PATH, save_path + "config_initializer_used.json"
        )
        # sets up the initial conditions, initializer.get_initial_state() passed to runner
        initializer = smh_initializer_epoch_two(
            INITIALIZER_CONFIG_PATH, GLOBAL_CONFIG_PATH
        )
        runner = MechanisticRunner(seip_ode)
        initial_state = initializer.get_initial_state()
        inferer = SMHInferer(
            GLOBAL_CONFIG_PATH, INFERER_CONFIG_PATH, runner, initial_state
        )

        hosp_data_filename = "%s_hospitalization.csv" % (
            initializer.config.REGIONS[0].replace(" ", "_")
        )
        hosp_data_path = os.path.join(
            inferer.config.HOSP_PATH, hosp_data_filename
        )
        hosp_data = pd.read_csv(hosp_data_path)
        hosp_data["date"] = pd.to_datetime(hosp_data["date"])
        if state == "HI":
            hosp_data.loc[
                (hosp_data["date"] > pd.to_datetime("2022-08-01"))
                & (hosp_data["agegroup"] == "0-17"),
                "hosp",
            ] = np.nan
        # align hosp to infections assuming 7-day inf -> hosp delay
        hosp_data["day"] = (
            hosp_data["date"] - pd.to_datetime(inferer.config.INIT_DATE)
        ).dt.days - 7
        # only keep hosp data that aligns to our initial date
        # sort ascending
        hosp_data = hosp_data.loc[
            (hosp_data["day"] >= 0) & (hosp_data["day"] <= model_day)
        ].sort_values(by=["day", "agegroup"], ascending=True, inplace=False)
        # make hosp into day x agegroup matrix
        obs_hosps = hosp_data.groupby(["day"])["hosp"].apply(np.array)
        obs_hosps_days = obs_hosps.index.to_list()
        obs_hosps = jnp.array(obs_hosps.to_list())

        sero_data_filename = "%s_sero.csv" % (
            initializer.config.REGIONS[0].replace(" ", "_")
        )
        sero_data_path = os.path.join(
            inferer.config.SERO_PATH, sero_data_filename
        )
        sero_data = pd.read_csv(sero_data_path)
        sero_data["date"] = pd.to_datetime(sero_data["date"])
        # align sero to infections assuming 14-day seroconversion delay
        sero_data["day"] = (
            sero_data["date"] - pd.to_datetime(inferer.config.INIT_DATE)
        ).dt.days - 14
        sero_data = sero_data.loc[
            (sero_data["day"] >= 0) & (sero_data["day"] <= model_day)
        ].sort_values(by=["day", "age"], ascending=True, inplace=False)
        # transform data to logit scale
        sero_data["logit_rate"] = np.log(
            sero_data["rate"] / (100.0 - sero_data["rate"])
        )
        # make sero into day x agegroup matrix
        obs_sero_lmean = sero_data.groupby(["day"])["logit_rate"].apply(
            np.array
        )
        obs_sero_days = obs_sero_lmean.index.to_list()
        obs_sero_lmean = jnp.array(obs_sero_lmean.to_list())
        obs_sero_lmean = obs_sero_lmean.at[np.isinf(obs_sero_lmean)].set(
            jnp.nan
        )
        # set sero sd, currently this is an arbitrary tunable parameters
        # dependent on sero sample size.
        obs_sero_n = sero_data.groupby(["day"])["n"].apply(np.array)
        obs_sero_lsd = 1.0 / jnp.sqrt(jnp.array(obs_sero_n.to_list()))
        obs_sero_lsd = obs_sero_lsd.at[jnp.isnan(obs_sero_lsd)].set(0.5)

        var_data_filename = "%s_strain_prop.csv" % (
            initializer.config.REGIONS[0].replace(" ", "_")
        )
        var_data_path = os.path.join(
            inferer.config.VAR_PATH, var_data_filename
        )
        # currently working up to third strain which is XBB1
        var_data = pd.read_csv(var_data_path)
        var_data = var_data[var_data["strain"] >= 2]
        var_data["date"] = pd.to_datetime(var_data["date"])
        var_data["day"] = (
            var_data["date"] - pd.to_datetime(inferer.config.INIT_DATE)
        ).dt.days  # no shift in alignment for variants
        var_data = var_data.loc[
            (var_data["day"] >= 0) & (var_data["day"] <= model_day)
        ].sort_values(by=["day", "strain"], ascending=True, inplace=False)
        obs_var_prop = var_data.groupby(["day"])["share"].apply(np.array)
        obs_var_days = obs_var_prop.index.to_list()
        obs_var_prop = jnp.array(obs_var_prop.to_list())
        # renormalizing the var prop
        obs_var_prop = obs_var_prop / jnp.sum(obs_var_prop, axis=1)[:, None]
        adj_factor = np.sqrt(np.sum(inferer.config.POPULATION))
        adj_factor = adj_factor if adj_factor >= np.sqrt(5e6) else np.sqrt(5e6)
        obs_var_sd = 90 / adj_factor

        inferer.infer(
            obs_hosps,
            obs_hosps_days,
            obs_sero_lmean,
            obs_sero_lsd,
            obs_sero_days,
            obs_var_prop,
            obs_var_days,
            obs_var_sd,
        )
        # plot the 4 compartments summed across all age bins and immunity status
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        print("Saving posterior samples to %s" % save_path)
        # saves all posterior samples including deterministic parameters
        inferer.checkpoint(save_path + "checkpoint.json", group_by_chain=True)
    except Exception as e:
        raise e
    finally:
        # bringing control back to the normal stdout and stderror
        sys.stdout = stdout
        sys.stderr = stderror
        stdout_file.close()
        stderror_file.close()
