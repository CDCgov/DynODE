from exp.fifty_state_5strain_2202_2404.postaz_process import (
    retrieve_inferer_obs,
    retrieve_post_samp,
    replace_and_simulate,
    simulate_hospitalization,
)
import arviz as az
from sklearn.metrics import mean_squared_error
import numpy as np
import jax.numpy as jnp
import jax
from jax import vmap, jit
from functools import partial
import random
import numpy as np
import pandas as pd
import numpyro.distributions as dist

# from numpyro import log_prob
import os
import numpyro
import multiprocessing as mp

# from exp.fifty_state_5strain_2202_2404.inferer_smh import SMHInferer


def hosp_var_posterior(
    samp, inferer, runner, particles_per_chain
):  # , initial_model_day=0):
    nsamp = len(samp["ihr_3"][0])  # Number of samples per chain
    nchain = len(samp["ihr_3"])  # Number of chains
    # each element of the list all_samples is a dictionary of unique values for each parameter from the ODE.
    # Randomly select a subset of posterior samples for simulation
    ranindex = random.sample(
        list(range(nsamp)), particles_per_chain
    )  # Randomly pick some samples per chain
    all_samples = [
        {k: v[c][r] for k, v in samp.items()} for r in ranindex for c in range(nchain)
    ]
    pred_hosps_list = []
    pred_var_list = []

    #     def process_per_sample(inferer, runner, f):
    #         output = replace_and_simulate(inferer, runner, f)
    #         ihr = jnp.array(
    #             [
    #                 f["ihr_mult_0"] * f["ihr_3"],
    #                 f["ihr_mult_1"] * f["ihr_3"],
    #                 f["ihr_mult_2"] * f["ihr_3"],
    #                 f["ihr_3"],
    #             ]
    #         )
    #         ihr_immune_mult = f["ihr_immune_mult"]
    #         ihr_jn1_mult = f["ihr_jn1_mult"]
    #         pred_hosps = simulate_hospitalization(
    #             output, ihr, ihr_immune_mult, ihr_jn1_mult
    #         )

    #         strain_incidence = jnp.sum(
    #             output.ys[inferer.config.COMPARTMENT_IDX.C],
    #             axis=(
    #                 inferer.config.I_AXIS_IDX.age + 1,
    #                 inferer.config.I_AXIS_IDX.hist + 1,
    #                 inferer.config.I_AXIS_IDX.vax + 1,
    #             ),
    #         )
    #         strain_incidence = jnp.diff(strain_incidence, axis=0)
    #         pred_vars = strain_incidence / jnp.sum(strain_incidence, axis=-1)

    #         return pred_hosps, pred_vars

    #     # Vectorize the function
    #     vectorized_process_sample = vmap(partial(process_per_sample, inferer, runner))

    #     # JIT compile the vectorized function
    #     jit_vectorized_process_sample = jit(vectorized_process_sample)

    #     # Convert the list of dictionaries to a dictionary of lists (if not already)
    #     all_samples_dict = {
    #         key: jnp.array(
    #             [f[key] for f in all_samples]
    #         ).flatten()  # here, f is a dictionary from all_samples, so we're varying over all dictionaries of all_samples which have a fixed key
    #         for key, balue in all_samples[0].items()
    #     }

    #     # Apply the JIT-compiled and vectorized function
    #     pred_hosps_array, pred_vars_array = jit_vectorized_process_sample(all_samples_dict)

    #     # Convert the arrays to lists if needed
    #     pred_hosps_list = list(pred_hosps_array)
    #     pred_var_list = list(pred_vars_array)

    #     return pred_hosps_list, pred_var_list

    for f in all_samples:
        output = replace_and_simulate(inferer, runner, f)
        ihr = jnp.array(
            [
                f["ihr_mult_0"] * f["ihr_3"],
                f["ihr_mult_1"] * f["ihr_3"],
                f["ihr_mult_2"] * f["ihr_3"],
                f["ihr_3"],
            ]
        )
        ihr_immune_mult = f["ihr_immune_mult"]
        ihr_jn1_mult = f["ihr_jn1_mult"]
        pred_hosps = simulate_hospitalization(output, ihr, ihr_immune_mult)

        strain_incidence = jnp.sum(
            output.ys[inferer.config.COMPARTMENT_IDX.C],
            axis=(
                inferer.config.I_AXIS_IDX.age + 1,
                inferer.config.I_AXIS_IDX.hist + 1,
                inferer.config.I_AXIS_IDX.vax + 1,
            ),
        )
        strain_incidence = jnp.diff(strain_incidence, axis=0)
        pred_vars = strain_incidence / jnp.sum(strain_incidence, axis=-1, keepdims=True)

        pred_hosps_list.append(pred_hosps)
        pred_var_list.append(pred_vars)

    return pred_hosps_list, pred_var_list


def mcmc_accuracy_measures(state, particles_per_chain):

    samp, fitted_means = retrieve_post_samp(state)
    (
        inferer,
        runner,
        obs_hosps,
        obs_hosps_days,
        obs_sero_lmean,
        obs_sero_days,
        obs_var_prop,
        obs_var_days,
    ) = retrieve_inferer_obs(
        state
    )  # , initial_model_day=0)
    pred_hosps_list, pred_var_list = hosp_var_posterior(
        samp, inferer, runner, particles_per_chain
    )  # , initial_model_day=0

    log_likelihoods = []
    for pred_hosps in pred_hosps_list:
        # Ensure that pred_hosps and obs_hosps_interval have the same shape
        mask_incidence = ~jnp.isnan(obs_hosps)
        with numpyro.handlers.mask(mask=mask_incidence):
            log_likelihood = dist.Poisson(pred_hosps).log_prob(obs_hosps)
        # log_likelihoods.append(log_likelihood)
    # Stack log likelihoods to create a single array and sum over the time axis and age group (axis=0, 1)
    # log_like = jnp.sum(jnp.stack(log_likelihood), axis=(0, 1))
    nsamp = len(samp["ihr_3"][0])  # Number of samples per chain
    nchain = len(samp["ihr_3"])  # Number of chains
    # each element of the list all_samples is a dictionary of unique values for each parameter from the ODE.
    # Randomly select a subset of posterior samples for simulation
    ranindex = random.sample(
        list(range(nsamp)), particles_per_chain
    )  # Randomly pick some samples per chain
    all_samples = [
        {k: v[c][r] for k, v in samp.items()} for r in ranindex for c in range(nchain)
    ]
    trace_hosps = az.from_dict(
        posterior_predictive={"hospitalizations": pred_hosps_list},
        observed_data={"hospitalizations": obs_hosps},
        log_likelihood={"log likelihood:": log_likelihood},
        posteriors={
            str(key): jnp.array(
                [f[key] for f in all_samples]
            ).flatten()  # here, f is a dictionary from all_samples, so we're varying over all dictionaries of all_samples which have a fixed key
            for key, balue in all_samples[0].items()
        },
    )
    waic_hosps = az.waic(trace_hosps)

    rmse = {}
    rmse_hosp = np.sqrt(mean_squared_error(jnp.median(pred_hosps_list), obs_hosps))
    for strain in range(pred_var_list[0].shape[-1]):
        rmse[f"RMSE strain {strain}"] = np.sqrt(
            mean_squared_error(jnp.median(pred_var_list[:, :, strain]), obs_hosps)
        )
    df_rmse = pd.DataFrame(rmse)
    df_rmse["RMSE Hospitalizations"] = rmse_hosp

    df_waic = pd.DataFrame(waic_hosps)
    # df_waic["WAIC for hospitalizations"] = waic_hosps
    df_total = pd.concat([df_waic, df_rmse])

    return df_total


states = [
    "AL",
    # "AK",
    # "AZ",
    # "AR",
    # "CA",
    # "CO",
    # "CT",
    # "DE",
    # # "FL",
    # # "GA",
    # # "HI",
    # # "ID",
    # # "IL",
    # # "IN",
    # # "IA",
    # # "KS",
    # # "KY",
    # # "LA",
    # # "ME",
    # # "MD",
    # # "MA",
    # # "MI",
    "MN",
    # # "MS",
    # # "MO",
    # # "MT",
    # # "NE",
    # # "NV",
    # # "NH",
    # # "NJ",
    # # "NM",
    # # "NY",
    # # "NC",
    # # "ND",
    # # "OH",
    # # "OK",
    # # "OR",
    # # "PA",
    # # "RI",
    # # "SC",
    # # "SD",
    # # "TN",
    # # "TX",
    # # "UT",
    # # "VT",
    # # "VA",
    # # "WA",
    # # "WV",
    # # "WI",
    # # "WY",
]
states_omit = [
    "WV",
    "GA",
    "AL",
    "TX",
    "WI",
    "WY",
    "VT",
    "VA",
]
az_output_path = "/output/fifty_state_5strain_2202_2404/SMH_5strains_071624/"

for st in states:
    json_file = os.path.join(az_output_path, st, "checkpoint.json")
    if not os.path.exists(json_file):
        states_omit.append(st)
states = list(set(states).difference(set(states_omit)))
states.sort()
# states = ["US", "AK", "AL", "IL", "WA"]
# states = ["US"] + states
print(states)
# %%
pool = mp.Pool(5)
df_total = zip(
    *pool.map(
        mcmc_accuracy_measures(state=st, particles_per_chain=100),
        [st for st in states],
    )
)
pool.close()
pd.concat(df_total).to_csv(f"output/accuracy.csv", index=False)
