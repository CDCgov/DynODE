from exp.fifty_state_5strain_2202_2404.postaz_process import (
    retrieve_inferer_obs,
    retrieve_post_samp,
    replace_and_simulate,
    simulate_hospitalization,
)
import arviz as az

# from sklearn.metrics import mean_squared_error
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

from exp.fifty_state_5strain_2202_2404.inferer_smh import SMHInferer

jax.config.update("jax_enable_x64", True)


def mcmc_accuracy_measures(state, particles_per_chain):
    print("getting samples via json.load()")
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
    ) = retrieve_inferer_obs(state)
    print("got inferer")
    # get random particle_index
    particle_indexes = np.random.choice(
        range(inferer.config.INFERENCE_NUM_SAMPLES),
        particles_per_chain,
        replace=False,
    )
    # select the particle for each chain make a list of tuples
    chain_particle_pairs = [
        (chain, particle)
        for chain in range(inferer.config.INFERENCE_NUM_CHAINS)
        for particle in particle_indexes
    ]
    # create your big list of hosp and var timeseries
    pred_hosps_list = []
    pred_vars_list = []
    # get a solutiong dict of (chain, particle) keys leading to the solution objects and hosp timelines
    posteriors_solution_dct = inferer.load_posterior_particle(
        chain_particle_pairs, max(obs_hosps_days) + 1, samp
    )
    nchain = len(samp["ihr_3"])
    # go through each chain and particle and build up a digestible list of the hosp and var props
    for chain in range(nchain):
        pred_hosps_chain = []
        pred_vars_chain = []
        for particle_index in particle_indexes:
            sol_dct = posteriors_solution_dct[(chain, particle_index)]
            # get the solution object and the predicted hosp out of the solution dct

            output = sol_dct["solution"]
            hosps = sol_dct["hospitalizations"]
            strain_incidence = jnp.sum(
                output.ys[inferer.config.COMPARTMENT_IDX.C],
                axis=(
                    inferer.config.I_AXIS_IDX.age + 1,
                    inferer.config.I_AXIS_IDX.hist + 1,
                    inferer.config.I_AXIS_IDX.vax + 1,
                ),
            )
            strain_incidence = jnp.diff(strain_incidence, axis=0)
            pred_vars = strain_incidence / jnp.sum(
                strain_incidence, axis=-1, keepdims=True
            )
            # select only obs hosp days
            pred_hosps_chain.append(jnp.array(hosps)[jnp.array(obs_hosps_days), ...])
            pred_vars_chain.append(pred_vars)
        pred_hosps_list.append(pred_hosps_chain)
        pred_vars_list.append(pred_vars_chain)
    # pred_hosps_list.shape == (nchain, ranindex, time_series, age_group)
    # pred_var_list.shape == (nchain, ranindex, time_series, strains)
    # obs_hosps.shape == (112, 4)
    nchain = len(samp["ihr_3"])
    log_likelihood_array = []
    # go through each chain/sample and get the log likelihood of pred hosp given obs hosp
    print("starting log-likelihood setup")
    for pred_hosps_chain in pred_hosps_list:
        log_likelihood_chain = []
        for pred_hosps in pred_hosps_chain:
            # Ensure that pred_hosps and obs_hosps_interval have the same shape
            mask_incidence = ~jnp.isnan(obs_hosps)
            with numpyro.handlers.mask(mask=mask_incidence):
                log_likelihood = dist.Poisson(pred_hosps).log_prob(obs_hosps)
            log_likelihood_chain.append(log_likelihood)
        log_likelihood_array.append(log_likelihood_chain)

    # repeat obs_hosp chain * sample times so the shape matches the pred hosp arrays
    obs_hosps = jnp.tile(jnp.array(obs_hosps), (nchain, len(particle_indexes), 1, 1))
    # log_likelihood_array.shape == (nchains, ranindex, 112, 4)
    # pred_hosps_list.shape == (nchains, ranindex, 112, 4)
    # get the posterior values for each chain to pass to the posteriors
    # TODO, possible bug here if posteriors_selected has different ordering than pred_hosps_list
    posteriors_selected = {
        key: np.array(value)[:, particle_indexes] for key, value in samp.items()
    }

    trace_hosps = az.from_dict(
        posterior_predictive={"hospitalizations": pred_hosps_list},
        observed_data={"hospitalizations": obs_hosps},
        log_likelihood={"log likelihood:": log_likelihood_array},
        posteriors=posteriors_selected,
    )

    waic_hosps = az.waic(trace_hosps)
    print(waic_hosps)
    rmse = {}
    # rmse_hosp = np.sqrt(
    #     mean_squared_error(
    #         jnp.median(jnp.array(pred_hosps_list), axis=0,1)[
    #             jnp.array(obs_hosps_days), :
    #         ],
    #         jnp.array(obs_hosps),
    #     )
    # )
    # for strain in range(jnp.array(pred_var_list)[0].shape[-1]):
    #     rmse[f"RMSE strain {strain}"] = np.sqrt(
    #         mean_squared_error(
    #             jnp.median(
    #                 jnp.array(pred_var_list[:, jnp.array(obs_hosps_days), strain]),
    #                 axis=(0, 1),
    #             ),
    #             obs_hosps,
    #         )
    #     )
    df_rmse = pd.DataFrame(rmse)
    # df_rmse["RMSE Hospitalizations"] = rmse_hosp

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
        mcmc_accuracy_measures(state=st, particles_per_chain=2),
        [st for st in states],
    )
)
pool.close()
pd.concat(df_total).to_csv(f"output/accuracy.csv", index=False)
