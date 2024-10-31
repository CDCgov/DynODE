# ruff: noqa: E402
import argparse
import json
import os
import shutil
import sys

import arviz
import jax
import numpy as np
import numpyro
from numpyro.infer import MCMC, NUTS

# adding things to path since in a docker container pathing gets changed
sys.path.append("/app/")
sys.path.append("/input/exp/fit_season2_5strain_2202_2404/")
print(os.getcwd())
# sys.path.append(".")
# sys.path.append(os.getcwd())
import jax.numpy as jnp
import pandas as pd
from fall_virus_inferer import FallVirusInferer

from resp_ode import CovidSeroInitializer, MechanisticRunner
from resp_ode.model_odes.seip_model_flatten_immune_hist import seip_ode
from resp_ode.utils import combine_strains, combined_strains_mapping
from src.mechanistic_azure.abstract_azure_runner import AbstractAzureRunner

jax.config.update("jax_enable_x64", True)
CACHE_MATRIX_JOB_ID = None

def rework_initial_state(initial_state):
    """
    Take the original `initial_state` which is (4, 16, 3, 5), collapsing the
    strain 0 and 1 into 0, and collapsing wane 3 and 4 into 3, resulting in
    a 3-strain initial_state with (4, 8, 3, 4).
    """
    hist_map, strain_map = combined_strains_mapping(1, 0, 3)
    s_new_1 = combine_strains(
        initial_state[0], hist_map, strain_map, 3, strain_axis=False
    )[:, :, :, :]
    s_new_2 = jnp.ones((4, 8, 3, 4)) * s_new_1[:, :, :, :4]
    s_new = s_new_2.at[:, :, :, 3].add(s_new_1[:, :, :, 4])
    e_new = combine_strains(
        initial_state[1], hist_map, strain_map, 3, strain_axis=True
    )[:, :, :, :]
    i_new = combine_strains(
        initial_state[2], hist_map, strain_map, 3, strain_axis=True
    )[:, :, :, :]
    c_new = initial_state[3][:, :, :, :]
    concatenate_shp = list(e_new.shape)
    concatenate_shp[3] = 2
    e_new = jnp.concatenate((e_new, jnp.zeros(tuple(concatenate_shp))), axis=3)
    i_new = jnp.concatenate((i_new, jnp.zeros(tuple(concatenate_shp))), axis=3)
    c_new_shape = list(s_new.shape)
    c_new_shape.append(i_new.shape[3])
    c_new = jnp.zeros(tuple(c_new_shape))
    # c_new = jnp.concatenate((c_new, jnp.zeros(tuple(concatenate_shp))), axis=3)
    initial_state = (
        s_new[:, 0:6, ...],
        e_new[:, 0:6, ...],
        i_new[:, 0:6, ...],
        c_new[:, 0:6, ...],
    )
    return initial_state


def override_mcmc(inferer, inverse_mass_matrix):
    # default to max tree depth of 5 if not specified
    inferer.inference_algo = MCMC(
        NUTS(
            inferer.likelihood,
            dense_mass=True,
            max_tree_depth=10,
            inverse_mass_matrix=inverse_mass_matrix,
            init_strategy=numpyro.infer.init_to_median,
        ),
        num_warmup=inferer.config.INFERENCE_NUM_WARMUP,
        num_samples=inferer.config.INFERENCE_NUM_SAMPLES,
        num_chains=inferer.config.INFERENCE_NUM_CHAINS,
        progress_bar=inferer.config.INFERENCE_PROGRESS_BAR,
    )


def get_loo_elpd(
        inferer,
        obs_hosps,
        obs_hosps_days,
        obs_sero_lmean,
        obs_sero_lsd,
        obs_sero_days,
        obs_var_prop,
        obs_var_days,
        obs_var_sd,
        tf
):
    samples = inferer.inference_algo.get_samples(group_by_chain=True)

    posteriors_selected = {
        key: np.array(values)[:, np.arange(0, 1000, 4)]
        for key, values in samples.items()
    }

    ll = numpyro.infer.util.log_likelihood(
        inferer.likelihood,
        posterior_samples=posteriors_selected,
        obs_hosps=obs_hosps,
        obs_hosps_days=obs_hosps_days,
        obs_sero_lmean=obs_sero_lmean,
        obs_sero_lsd=obs_sero_lsd,
        obs_sero_days=obs_sero_days,
        obs_var_prop=obs_var_prop,
        obs_var_days=obs_var_days,
        obs_var_sd=obs_var_sd,
        tf=tf,
        parallel=True,
        batch_ndims=2,
    )

    ll_incd = arviz.from_dict(
        posterior=posteriors_selected,
        log_likelihood={"log_likelihood": ll["incidence"]},
    )
    ll_incd_loo = arviz.loo(ll_incd)
    ll_varprop = arviz.from_dict(
        posterior=posteriors_selected,
        log_likelihood={"log_likelihood": ll["variant_proportion"]},
    )
    ll_varprop_loo = arviz.loo(ll_varprop)
    loo_results = {
        "incidence_elpd_est": ll_incd_loo["elpd_loo"],
        "incidence_elpd_se": ll_incd_loo["se"],
        "incidence_warning": ll_incd_loo["warning"],
        "varprop_elpd_est": ll_varprop_loo["elpd_loo"],
        "varprop_elpd_se": ll_varprop_loo["se"],
        "varprop_warning": ll_varprop_loo["warning"],
    }

    return loo_results


def save_mass_matrix(inferer, save_path):
    last_adapt_state = getattr(inferer.inference_algo.last_state, "adapt_state")
    inverse_mass_matrix = getattr(last_adapt_state, "inverse_mass_matrix")
    reform_inverse_mass_matrix = {str(k): v.tolist() for k, v in inverse_mass_matrix.items()}
    json.dump(reform_inverse_mass_matrix, open(save_path, "w"))


def preprocess_observed_data(initializer, inferer, model_day):
    """A function responsible for reading in observed data, preprocessing it, and returning
    a tuple of timeseries

    Parameters
    ----------
    initializer : CovidSeroInitializer
        initializer used to construct initial states
    inferer : MechanisticInferer
        inferer used to calculate likelihood on this observed data
    model_day : int
        Number of days on which the model will run, and the observed data should span

    Returns
    -------
    tuple[Jax.Array]
        tuple of observed datasets, specifically hospitalization, serology, and variant proportion.
    """
    hosp_data_filename = "%s_hospitalization.csv" % (
        initializer.config.REGIONS[0].replace(" ", "_")
    )
    hosp_data_path = os.path.join(inferer.config.HOSP_PATH, hosp_data_filename)
    hosp_data = pd.read_csv(hosp_data_path)
    hosp_data["date"] = pd.to_datetime(hosp_data["date"])
    # special setting for HI
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
    sero_data_path = os.path.join(inferer.config.SERO_PATH, sero_data_filename)
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
    obs_sero_lmean = sero_data.groupby(["day"])["logit_rate"].apply(np.array)
    obs_sero_days = obs_sero_lmean.index.to_list()
    obs_sero_lmean = jnp.array(obs_sero_lmean.to_list())
    obs_sero_lmean = obs_sero_lmean.at[np.isinf(obs_sero_lmean)].set(jnp.nan)
    # set sero sd, currently this is an arbitrary tunable parameters
    # dependent on sero sample size.
    obs_sero_n = sero_data.groupby(["day"])["n"].apply(np.array)
    obs_sero_lsd = 1.0 / jnp.sqrt(jnp.array(obs_sero_n.to_list()))
    obs_sero_lsd = obs_sero_lsd.at[jnp.isnan(obs_sero_lsd)].set(0.5)

    var_data_filename = "%s_strain_prop.csv" % (
        initializer.config.REGIONS[0].replace(" ", "_")
    )
    var_data_path = os.path.join(inferer.config.VAR_PATH, var_data_filename)
    # currently working up to third strain which is XBB1
    var_data = pd.read_csv(var_data_path)
    var_data = var_data[var_data["strain"] < 5]
    var_data["date"] = pd.to_datetime(var_data["date"])
    var_data["day"] = (
        var_data["date"] - pd.to_datetime("2022-02-11")
    ).dt.days  # no shift in alignment for variants
    var_data = var_data.loc[
        (var_data["day"] >= 0) & (var_data["day"] <= model_day)
    ].sort_values(by=["day", "strain"], ascending=True, inplace=False)
    obs_var_prop = var_data.groupby(["day"])["share"].apply(np.array)
    obs_var_days = obs_var_prop.index.to_list()
    obs_var_prop = jnp.array(obs_var_prop.to_list())
    # renormalizing the var prop
    obs_var_prop = obs_var_prop / jnp.sum(obs_var_prop, axis=1)[:, None]
    obs_var_sd = 150 / jnp.sqrt(jnp.sum(inferer.config.POPULATION))
    return (
        obs_hosps,
        obs_hosps_days,
        obs_sero_lmean,
        obs_sero_lsd,
        obs_sero_days,
        obs_var_prop,
        obs_var_days,
        obs_var_sd,
    )


class EpochOneRunner(AbstractAzureRunner):
    # __init__ already implemented by the abstract case
    def __init__(self, azure_output_dir):
        super().__init__(azure_output_dir)

    def process_state(self, state, jobid=None, jobid_in_path=False):
        model_day = 800
        # step 1: define your paths, now in the input
        state_config_path = os.path.join(
            "/input/exp/fifty_state_season2_5strain_2202_2404/states",
            state,
        )
        if jobid_in_path:
            state_config_path = os.path.join(
                "/input/exp/fifty_state_season2_5strain_2202_2404",
                jobid,
                "states",
                state,
            )
        # state_config_path = "exp/fifty_state_sero_second_try/" + args.state + "/"
        print("Running the following state: " + state + "\n")
        # global_config include definitions such as age bin bounds and strain definitions
        # Any value or data structure that needs context to be interpretted is here.
        GLOBAL_CONFIG_PATH = os.path.join(
            state_config_path, "config_global.json"
        )
        # a temporary global config that matches with original initializer
        TEMP_GLOBAL_CONFIG_PATH = os.path.join(
            state_config_path, "temp_config_global.json"
        )
        global_js = json.load(open(GLOBAL_CONFIG_PATH))
        global_js["NUM_STRAINS"] = 3
        global_js["NUM_WANING_COMPARTMENTS"] = 5
        global_js["WANING_TIMES"] = [70, 70, 70, 129, 0]
        json.dump(global_js, open(TEMP_GLOBAL_CONFIG_PATH, "w"))

        # defines the init conditions of the scenario: pop size, initial infections etc.
        INITIALIZER_CONFIG_PATH = os.path.join(
            state_config_path, "config_initializer.json"
        )
        # defines prior __distributions__ for inferring runner variables.
        INFERER_CONFIG_PATH = os.path.join(
            state_config_path, "config_inferer.json"
        )
        # save copies of the used config files to output for reproducibility purposes
        cg_path = self.azure_output_dir + "config_global_used.json"
        cinf_path = self.azure_output_dir + "config_inferer_used.json"
        cini_path = self.azure_output_dir + "config_initializer_used.json"
        if os.path.exists(cg_path):
            os.remove(cg_path)
        if os.path.exists(cinf_path):
            os.remove(cinf_path)
        if os.path.exists(cini_path):
            os.remove(cini_path)
        shutil.copy(GLOBAL_CONFIG_PATH, cg_path)
        shutil.copy(INFERER_CONFIG_PATH, cinf_path)
        shutil.copy(INITIALIZER_CONFIG_PATH, cini_path)
        # sets up the initial conditions, initializer.get_initial_state() passed to runner
        initializer = CovidSeroInitializer(
            INITIALIZER_CONFIG_PATH, TEMP_GLOBAL_CONFIG_PATH
        )
        runner = MechanisticRunner(seip_ode)
        initial_state = initializer.get_initial_state()
        initial_state = rework_initial_state(initial_state)
        inferer = FallVirusInferer(
            GLOBAL_CONFIG_PATH, INFERER_CONFIG_PATH, runner, initial_state
        )
        # Check if there's cache matrix
        if CACHE_MATRIX_JOB_ID:
            matrix_path = os.path.join("/output/fifty_state_season2_5strain_2202_2404", CACHE_MATRIX_JOB_ID, state, "inverse_mass_matrix.json")
            if os.path.exists(matrix_path):
                inverse_mass_matrix_json = json.load(open(matrix_path))
                inverse_mass_matrix = {eval(k): jnp.mean(jnp.array(v), axis=0) for k, v in inverse_mass_matrix_json.items()}
                print([v.shape for k, v in inverse_mass_matrix.items()])
                override_mcmc(inferer, inverse_mass_matrix)
            else:
                print("Couldn't find corresponding inverse_mass_matrix, initializing at default.")

        (
            obs_hosps,
            obs_hosps_days,
            obs_sero_lmean,
            obs_sero_lsd,
            obs_sero_days,
            obs_var_prop,
            obs_var_days,
            obs_var_sd,
        ) = preprocess_observed_data(initializer, inferer, model_day)

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
        # saves all posterior samples including deterministic parameters
        self.save_inference_posteriors(inferer)
        self.save_inference_final_timesteps(inferer)
        self.save_inference_timelines(inferer)
        save_mass_matrix(inferer, os.path.join(self.azure_output_dir, "inverse_mass_matrix.json"))

        loo_results = get_loo_elpd(
            inferer,
            obs_hosps,
            obs_hosps_days,
            obs_sero_lmean,
            obs_sero_lsd,
            obs_sero_days,
            obs_var_prop,
            obs_var_days,
            obs_var_sd,
            max(obs_hosps_days) + 1
        )
        save_path = os.path.join(self.azure_output_dir, "loo_results.json")
        json.dump(loo_results, open(save_path, "w"))

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
parser.add_argument(
    "-l", "--local", action="store_false", help="scenario being run on Azure"
)

if __name__ == "__main__":
    args = parser.parse_args()
    jobid = args.jobid
    state = args.state
    local = args.local
    save_path = "/output/fifty_state_season2_5strain_2202_2404/%s/%s/" % (
        jobid,
        state,
    )
    runner = EpochOneRunner(save_path)
    runner.process_state(state, jobid, jobid_in_path=local)
