from exp.fifty_state_6strain_2202_2407.postaz_process import (
    retrieve_inferer_obs,
    retrieve_post_samp,
)
import arviz as az

from sklearn.metrics import mean_squared_error
import numpy as np
import jax.numpy as jnp
import jax
import random
import numpy as np
import pandas as pd
import numpyro.distributions as dist
import os
import numpyro

# from mechanistic_model import mechanistic_inferer
# from mechanistic_model.mechanistic_inferer import load_posterior_particle

# import multiprocessing as mp
from exp.fifty_state_6strain_2202_2407.inferer_smh import SMHInferer

jax.config.update("jax_enable_x64", True)


def mcmc_accuracy_measures(
    state,
    particles_per_chain,
    initial_model_day,
    az_output,
    ic,
    variant=False,
):
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
    ) = retrieve_inferer_obs(state, initial_model_day)
    print("obs_var_prop and hosps shapes: ", obs_var_prop.shape, obs_hosps.shape)
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
            pred_vars_chain.append(jnp.array(pred_vars)[jnp.array(obs_var_days), ...])
        print(jnp.shape(jnp.array(hosps)), jnp.shape(jnp.array(pred_vars)))
        pred_hosps_list.append(pred_hosps_chain)
        pred_vars_list.append(pred_vars_chain)

        # pred_hosps_list.shape == (nchain, ranindex, time_series, age_group)
        # pred_var_list.shape == (nchain, ranindex, time_series, strains)
        # obs_hosps.shape == (112, 4)
    nchain = len(samp["ihr_3"])
    log_likelihood_array_hosps = []
    # go through each chain/sample and get the log likelihood of pred hosp given obs hosp
    print("starting log-likelihood setup")
    for pred_hosps_chain in pred_hosps_list:
        log_likelihood_chain_hosps = []
        for pred_hosps in pred_hosps_chain:
            # Ensure that pred_hosps and obs_hosps_interval have the same shape
            mask_incidence = ~jnp.isnan(obs_hosps)
            with numpyro.handlers.mask(mask=mask_incidence):
                log_likelihood = dist.Poisson(pred_hosps).log_prob(obs_hosps)
                log_likelihood_chain_hosps.append(log_likelihood)
        log_likelihood_array_hosps.append(log_likelihood_chain_hosps)

        # repeat obs_hosp, obs_var_prop chain * sample times so the shape matches the pred hosp arrays
    obs_hosps = jnp.tile(jnp.array(obs_hosps), (nchain, len(particle_indexes), 1, 1))

    # log_likelihood_array.shape == (nchains, ranindex, 112, 4)
    # pred_hosps_list.shape == (nchains, ranindex, 112, 4)
    # get the posterior values for each chain to pass to the posteriors
    # TODO, possible bug here if posteriors_selected has different ordering than pred_hosps_list
    posteriors_selected = {
        key: np.array(value)[:, particle_indexes] for key, value in samp.items()
    }

    trace_hosps = az.from_dict(
        posterior=posteriors_selected,
        posterior_predictive={"hospitalizations": pred_hosps_list},
        observed_data={"hospitalizations": obs_hosps},
        log_likelihood={"log likelihood:": log_likelihood_array_hosps},
    )
    if ic == "waic":
        waic_hosps = az.waic(trace_hosps)
        df_waic_hosps = pd.DataFrame(waic_hosps)
        df_waic_hosps.index = [x + f"_hosps" for x in df_waic_hosps.index]

        if variant == True:
            # obs_var_prop = jnp.tile(
            #     jnp.array(obs_var_prop), (nchain, len(particle_indexes), 1, 1)
            # )
            print(jnp.shape(jnp.array(obs_var_prop)))
            log_likelihood_array_vars = []
            for pred_vars_chain in pred_vars_list:
                log_likelihood_chain_vars = []
                for pred_vars in pred_vars_chain:
                    mask_incidence = ~jnp.isnan(obs_var_prop)
                    with numpyro.handlers.mask(mask=mask_incidence):
                        pred_vars_sd = jnp.ones(
                            jnp.shape(jnp.array(pred_vars))
                        ) * jnp.std(jnp.array(obs_var_prop))
                        log_likelihood = dist.Normal(pred_vars, pred_vars_sd).log_prob(
                            obs_var_prop
                        )
                    log_likelihood_chain_vars.append(log_likelihood)
                log_likelihood_array_vars.append(log_likelihood_chain_vars)
            trace_hosps = az.from_dict(
                posterior=posteriors_selected,
                posterior_predictive={"pred_vars_prop": pred_vars_list},
                observed_data={"vars_prop_obs_data": obs_var_prop},
                log_likelihood={"log likelihood:": log_likelihood_array_vars},
            )
            waic_vars = az.waic(trace_hosps)
            df_waic_vars = pd.DataFrame(waic_vars)
            df_waic_vars.drop(["waic_i"], axis=0, inplace=True)
            df_waic_vars.index = [
                x + f"_variant_proportions" for x in df_waic_vars.index
            ]

            df_waic = pd.concat([df_waic_hosps, df_waic_vars], axis=0)
            df_waic.columns = [state]
            return df_waic, waic_hosps, waic_vars
        else:
            return df_waic_hosps, waic_hosps
    if ic == "loo":
        trace_hosps = az.from_dict(
            posterior={"param": posteriors_selected},
            posterior_predictive={"hospitalizations": pred_hosps_list},
            observed_data={"hospitalizations": obs_hosps},
            log_likelihood={"log_likelihood": log_likelihood_array_hosps},
        )
        loo_hosps = az.loo(trace_hosps)
        df_loo_hosps = pd.DataFrame(loo_hosps)

        df_loo_hosps.index = [x + f"_hosps" for x in df_loo_hosps.index]

        if variant == True:
            # obs_var_prop = jnp.tile(
            #     jnp.array(obs_var_prop), (nchain, len(particle_indexes), 1, 1)
            # )
            log_likelihood_array_vars = []
            for pred_vars_chain in pred_vars_list:
                log_likelihood_chain_vars = []
                for pred_vars in pred_vars_chain:
                    mask_incidence = ~jnp.isnan(obs_var_prop)
                    with numpyro.handlers.mask(mask=mask_incidence):
                        pred_vars_sd = jnp.ones(
                            jnp.shape(jnp.array(pred_vars))
                        ) * jnp.std(jnp.array(obs_var_prop))
                        log_likelihood = dist.Normal(pred_vars, pred_vars_sd).log_prob(
                            obs_var_prop
                        )
                    log_likelihood_chain_vars.append(log_likelihood)
                log_likelihood_array_vars.append(log_likelihood_chain_vars)
            trace_vars = az.from_dict(
                posterior={"param": posteriors_selected},
                posterior_predictive={"pred_vars_prop": pred_vars_list},
                observed_data={"vars_prop_obs_data": obs_var_prop},
                log_likelihood={"log likelihood:": log_likelihood_array_vars},
            )
            loo_vars = az.loo(trace_vars)
            df_loo_vars = pd.DataFrame(loo_vars)
            df_loo_vars.index = [x + f"_variant_proportions" for x in df_loo_vars.index]

            df_loo = pd.concat([df_loo_hosps, df_loo_vars], axis=0)
            df_loo.columns = [state]
            return df_loo, loo_hosps, loo_vars
        else:
            return df_loo_hosps, loo_hosps


if __name__ == "__main__":
    states = [
        "AL",
        "AK",
        # "AZ",
        # "AR",
        "CA",
        # "CO",
        # "CT",
        # "DE",
        # "FL",
        # "GA",
        # "HI",
        # "ID",
        # "IL",
        # "IN",
        # "IA",
        # "KS",
        # "KY",
        # "LA",
        # "ME",
        # "MD",
        # "MA",
        # "MI",
        # "MN",
        # "MS",
        # "MO",
        # "MT",
        # "NE",
        # "NV",
        # "NH",
        # "NJ",
        # "NM",
        # "NY",
        # "NC",
        # "ND",
        # "OH",
        # "OK",
        # "OR",
        # "PA",
        # "RI",
        # "SC",
        # "SD",
        # "TN",
        # "TX",
        # "UT",
        # "VT",
        # "VA",
        # "WA",
        # "WV",
        # "WI",
        # "WY",
    ]
    az_output_path = "/output/fifty_state_2204_2407_6strain/smh_6str_prelim_7/"
    suffix0 = "prelim_7_waic"
    # dframe = {}
    # for st in states:
    #     try:
    #         print(f"Processing state: {st}")
    #         result, hosps, vars = mcmc_accuracy_measures(
    #             state=st,
    #             particles_per_chain=80,
    #             initial_model_day=560,
    #             az_output=az_output_path,
    #             ic="waic",
    #             variant=False,
    #         )
    #         print(f"Result for state {st}:")
    #         print(result)
    #         dframe[st] = result
    #     except Exception as e:
    #         print(f"Error processing state {st}: {e}")

    #     # Combine individual DataFrames into oneif dframe:
    #     final_df = pd.concat(dframe.values(), axis=1)
    #     final_df.columns = dframe.keys()
    #     print(final_df)

    # final_df.to_csv(f"output/accuracy{suffix0}.csv", index=True)

    suffix1 = "prelim_7_waic_w/_vars"
    dframe = {}
    for st in states:
        try:
            print(f"Processing state: {st}")
            result, hosps, vars = mcmc_accuracy_measures(
                state=st,
                particles_per_chain=5,
                initial_model_day=560,
                az_output=az_output_path,
                ic="waic",
                variant=True,
            )
            print(f"Result for state {st}:")
            print(result)
            dframe[st] = result
        except Exception as e:
            print(f"Error processing state {st}: {e}")

        # Combine individual DataFrames into one dframe:
        final_df = pd.concat(dframe.values(), axis=1)
        final_df.columns = dframe.keys()
        print(final_df)

    final_df.to_csv(f"output/accuracy{suffix1}.csv", index=True)
