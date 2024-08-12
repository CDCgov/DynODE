from exp.fifty_state_6strain_2202_2407.postaz_process import (
    retrieve_inferer_obs,
    retrieve_post_samp,
)
from scipy import stats
import arviz as az
import argparse
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

from mechanistic_model import mechanistic_inferer
from mechanistic_model.mechanistic_inferer import load_posterior_particle

# import multiprocessing as mp
from exp.fifty_state_6strain_2202_2407.inferer_smh import SMHInferer

jax.config.update("jax_enable_x64", True)


# Parsing states as arguments
# def parse_arguments():
#     parser = argparse.ArgumentParser(
#         description="Run MCMC accuracy measures for multiple states."
#     )
#     parser.add_argument(
#         "-s",
#         "--states",
#         type=str,
#         required=True,
#         nargs="+",
#         help="Space-separated list of USPS postal codes representing the states",
#     )
#     return parser.parse_args()


# Loads the particles and computes the log-likelihoods.
def load_and_compute_likelihoods(
    state, particles_per_chain, initial_model_day, variant=False
):
    # Load samples and initialize inferer and observation data
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

    # Select random particles
    particle_indexes = np.random.choice(
        range(inferer.config.INFERENCE_NUM_SAMPLES),
        particles_per_chain,
        replace=False,
    )
    chain_particle_pairs = [
        (chain, particle)
        for chain in range(inferer.config.INFERENCE_NUM_CHAINS)
        for particle in particle_indexes
    ]

    # Load posterior particles
    posteriors_solution_dct = inferer.load_posterior_particle(
        chain_particle_pairs, max(obs_hosps_days) + 1, samp
    )

    nchain = len(samp["ihr_3"])
    pred_hosps_list = []
    pred_vars_list = []

    # Compute hospitalizations and variant proportions predictions
    for chain in range(nchain):
        pred_hosps_chain = []
        pred_vars_chain = []
        for particle_index in particle_indexes:
            sol_dct = posteriors_solution_dct[(chain, particle_index)]
            output = sol_dct["solution"]
            hosps = sol_dct["hospitalizations"]
            if variant:
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
            pred_hosps_chain.append(jnp.array(hosps)[jnp.array(obs_hosps_days), ...])
            pred_vars_chain.append(jnp.array(pred_vars)[jnp.array(obs_var_days), ...])

        pred_hosps_list.append(pred_hosps_chain)
        pred_vars_list.append(pred_vars_chain)

    # Compute log-likelihood for hospitalizations
    log_likelihood_array_hosps = []
    for pred_hosps_chain in pred_hosps_list:
        log_likelihood_chain_hosps = []
        for pred_hosps in pred_hosps_chain:
            mask_incidence = ~jnp.isnan(obs_hosps)
            with numpyro.handlers.mask(mask=mask_incidence):
                log_likelihood = dist.Poisson(pred_hosps).log_prob(obs_hosps)
                log_likelihood_chain_hosps.append(log_likelihood)
        log_likelihood_array_hosps.append(log_likelihood_chain_hosps)

    obs_hosps = jnp.tile(jnp.array(obs_hosps), (nchain, len(particle_indexes), 1, 1))

    result = {
        "obs_hosps": obs_hosps,
        "pred_hosps_list": pred_hosps_list,
        "log_likelihood_array_hosps": log_likelihood_array_hosps,
        "posteriors_selected": {
            key: np.array(value)[:, particle_indexes] for key, value in samp.items()
        },
    }

    if variant:
        log_likelihood_array_vars = []
        for pred_vars_chain in pred_vars_list:
            log_likelihood_chain_vars = []
            for pred_vars in pred_vars_chain:
                mask_incidence = ~jnp.isnan(obs_var_prop)
                with numpyro.handlers.mask(mask=mask_incidence):
                    pred_vars_sd = jnp.ones(jnp.shape(jnp.array(pred_vars))) * jnp.std(
                        jnp.array(obs_var_prop)
                    )
                    log_likelihood = dist.Normal(pred_vars, pred_vars_sd).log_prob(
                        obs_var_prop
                    )
                log_likelihood_chain_vars.append(log_likelihood)
            log_likelihood_array_vars.append(log_likelihood_chain_vars)

        result["obs_var_prop"] = obs_var_prop
        result["pred_vars_list"] = pred_vars_list
        result["log_likelihood_array_vars"] = log_likelihood_array_vars

    return result


# This function will create the ArviZ trace and compute either WAIC or LOO based on the input parameters. It uses the output of the first function.
def create_and_compute_ic(data, ic, state, variant=False):
    trace_hosps = az.from_dict(
        posterior=data["posteriors_selected"],
        posterior_predictive={"hospitalizations": data["pred_hosps_list"]},
        observed_data={"hospitalizations": data["obs_hosps"]},
        log_likelihood={"log_likelihood": data["log_likelihood_array_hosps"]},
    )

    if ic == "waic":
        waic_hosps = az.waic(trace_hosps)
        if variant:
            trace_vars = az.from_dict(
                posterior=data["posteriors_selected"],
                posterior_predictive={"pred_vars_prop": data["pred_vars_list"]},
                observed_data={"vars_prop_obs_data": data["obs_var_prop"]},
                log_likelihood={"log_likelihood": data["log_likelihood_array_vars"]},
            )
            waic_vars = az.waic(trace_vars)
            return [waic_hosps, waic_vars]
        else:
            return [waic_hosps]

    elif ic == "loo":
        loo_hosps = az.loo(trace_hosps)
        if variant:
            trace_vars = az.from_dict(
                posterior=data["posteriors_selected"],
                posterior_predictive={"pred_vars_prop": data["pred_vars_list"]},
                observed_data={"vars_prop_obs_data": data["obs_var_prop"]},
                log_likelihood={"log_likelihood": data["log_likelihood_array_vars"]},
            )
            loo_vars = az.loo(trace_vars)
            return [loo_hosps, loo_vars]
        else:
            return [loo_hosps]

    else:
        raise ValueError("Invalid information criterion (ic) specified.")


def mcmc_accuracy_measures(
    state, particles_per_chain, initial_model_day, az_output, ic, variant=False
):
    data = load_and_compute_likelihoods(
        state, particles_per_chain, initial_model_day, variant
    )
    result = create_and_compute_ic(data, ic, state, variant)
    return result


def comparison_per_state(
    state, particles_per_chain, initial_model_day, az_outputs_list, ic, variant
):
    # Initialize dictionaries to hold the model results
    compare_dict_hosps = {}
    compare_dict_vars = {}

    # Iterate through each model output in the az_outputs_list
    for k, az_output in enumerate(az_outputs_list):
        hosps = mcmc_accuracy_measures(
            state=state,
            particles_per_chain=particles_per_chain,
            initial_model_day=initial_model_day,
            az_output=az_output,
            ic=ic,
            variant=variant,
        )[0]
        # Add the results to the comparison dictionaries
        compare_dict_hosps[f"hospitalizations_model_{k+1}"] = hosps
        if variant:
            compare_dict_vars[f"variant_props_model_{k+1}"] = mcmc_accuracy_measures(
                state=state,
                particles_per_chain=particles_per_chain,
                initial_model_day=initial_model_day,
                az_output=az_output,
                ic=ic,
                variant=variant,
            )[1]

    # Perform comparison for hospitalizations
    compare_df_hosps = az.compare(compare_dict_hosps)
    p_hosps = compare_df_hosps["elpd_diff"] / compare_df_hosps["dse"]
    compare_df_hosps["p_value"] = 2 * (1 - stats.norm.cdf(abs(p_hosps)))
    compare_df_hosps["state"] = [state] * len(compare_df_hosps)

    if variant:
        # Perform comparison for variant proportions
        compare_df_vars = az.compare(compare_dict_vars)
        p_vars = compare_df_vars["elpd_diff"] / compare_df_vars["dse"]
        compare_df_vars["p_value"] = 2 * (1 - stats.norm.cdf(abs(p_vars)))
        compare_df_vars["state"] = [state] * len(compare_df_vars)
        # Concatenate hospitalization and variant comparison DataFrames
        compare_df = pd.concat([compare_df_hosps, compare_df_vars], axis=0)
    else:
        # If variant is False, only return the hospitalization comparison DataFrame
        compare_df = compare_df_hosps

    return compare_df


# main_func() will return the concatenation of the previous function outputs for the desired states
def main(
    particles_per_chain,
    initial_model_day,
    ic,
    variant,
    az_outputs_list,
    output_csv_path,
):
    parser = argparse.ArgumentParser(
        description="Run the MCMC accuracy measures comparison for selected states."
    )
    parser.add_argument(
        "-s",
        "--states",
        type=str,
        required=True,
        nargs="+",
        help="space-separated list of USPS postal codes representing each state",
    )
    args = parser.parse_args()
    states = args.states

    all_states_df = []

    for state in states:
        compare_df = comparison_per_state(
            state, particles_per_chain, initial_model_day, az_outputs_list, ic, variant
        )
        all_states_df.append(compare_df)

    final_comparison_df = pd.concat(all_states_df, ignore_index=True)
    final_comparison_df.to_csv(output_csv_path, index=False)


# Example call to main function (replace with actual values)
main(
    particles_per_chain=10,
    initial_model_day=0,
    ic="loo",
    variant=False,
    az_outputs_list=[
        "/output/fifty_state_6strain_2204_2407/smh_6str_prelim_3/",
        "/output/fifty_state_6strain_2204_2407/smh_6str_prelim_4/",
        "/output/fifty_state_6strain_2204_2407/smh_6str_prelim_6/",
        "/output/fifty_state_6strain_2204_2407/smh_6str_prelim_7/",
    ],
    output_csv_path="comparison_prelims_results.csv",
)
