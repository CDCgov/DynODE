from exp.fifty_state_6strain_2202_2407.postaz_process import (
    retrieve_inferer_obs,
    retrieve_post_samp,
)

# exit()
import arviz as az
import numpy as np
import jax.numpy as jnp
import jax
import random
import numpy as np
import pandas as pd
import numpyro.distributions as dist
import os
import numpyro
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from exp.fifty_state_6strain_2202_2407.inferer_smh import SMHInferer

jax.config.update("jax_enable_x64", True)


# variant should be either set to True or False
def loglikelihood(
    state,
    particles_per_chain,
    initial_model_day,
    az_output_path,
    likelihood_poisson,
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
    for i, pred_hosps_chain in enumerate(pred_hosps_list):
        log_likelihood_chain_hosps = []
        for j, pred_hosps in enumerate(pred_hosps_chain):
            # Ensure that pred_hosps and obs_hosps_interval have the same shape
            mask_incidence = ~jnp.isnan(obs_hosps)
            if likelihood_poisson:
                with numpyro.handlers.mask(mask=mask_incidence):
                    log_likelihood = dist.Poisson(pred_hosps).log_prob(obs_hosps)
                log_likelihood_chain_hosps.append(log_likelihood)
            else:
                with numpyro.handlers.mask(mask=mask_incidence):
                    vmr = jnp.var(obs_hosps, axis=0) / jnp.mean(obs_hosps, axis=0)
                    mu = jnp.mean(obs_hosps, axis=0)
                    pos_vmr = jnp.maximum(
                        vmr - jnp.array([1] * obs_hosps.shape[1]),
                        jnp.array([0.01] * obs_hosps.shape[1]),
                    )
                    alpha = mu / pos_vmr
                    mult = samp["concentration_multiplier"][i][particle_indexes[j]]
                    conc = alpha / mult
                    log_likelihood = dist.NegativeBinomial2(
                        concentration=jnp.multiply(
                            conc, jnp.array([20] * obs_hosps.shape[1])
                        ),
                        mean=pred_hosps,
                    ).log_prob(obs_hosps)
                log_likelihood_chain_hosps.append(log_likelihood)
        log_likelihood_array_hosps.append(log_likelihood_chain_hosps)
    if variant:
        print(jnp.shape(jnp.array(obs_var_prop)))
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
            log_likelihood_array_vars.append(
                jnp.absolute(jnp.array(log_likelihood_chain_vars))
            )
            # log_likelihood_array.shape == (nchains, ranindex, 112, 4)
        # pred_hosps_list.shape == (nchains, ranindex, 112, 4)
        # get the posterior values for each chain to pass to the posteriors
        print("log_likelihood vars shape:", jnp.shape(log_likelihood_chain_vars))
        print("log_likelihood hosps shape:", jnp.shape(log_likelihood_chain_hosps))

        return jnp.array(log_likelihood_array_hosps), jnp.multiply(
            jnp.array(log_likelihood_array_vars), -1
        )
    else:
        return jnp.array(log_likelihood_array_hosps)


if __name__ == "__main__":
    states = [
        "AL",
        "AK",
        # # "AZ",
        # # "AR",
        # "CA",
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
    ratio = pd.DataFrame(
        columns=[
            "Poisson log_likelihood(Hosps)/log_likelihood(Var_Prop) ",
            "NB log_likelihood(Hosps)/log_likelihood(Var_Prop) ",
        ],
    )
    i = 0
    pdf_path = "output/hosps_vars_log_difference_all_states.pdf"
    with PdfPages(pdf_path) as pdf:
        for st in states:
            az_output_path_list = [
                "/output/fifty_state_2204_2407_6strain/ant-nbinomial",
                "/output/fifty_state_2204_2407_6strain/ant-fix_conc_more_var_prop",
            ]

            log_likelihood_poisson_hosps, log_likelihood_poisson_vars = loglikelihood(
                state=st,
                particles_per_chain=10,
                initial_model_day=560,
                az_output_path=az_output_path_list[0],
                likelihood_poisson=True,
                variant=True,
            )
            log_likelihood_nb_hosps, log_likelihood_nb_vars = loglikelihood(
                state=st,
                particles_per_chain=10,
                initial_model_day=560,
                az_output_path=az_output_path_list[1],
                likelihood_poisson=False,
                variant=True,
            )
            nb_hosps = jnp.sum(jnp.array(log_likelihood_nb_hosps), axis=(0, 1, 3))
            nb_vars = jnp.sum(jnp.array(log_likelihood_nb_vars), axis=(0, 1, 3))

            nb = jnp.sum(nb_hosps) + jnp.sum(nb_vars)
            poisson_hosps = jnp.sum(
                jnp.array(log_likelihood_poisson_hosps), axis=(0, 1, 3)
            )
            poisson_vars = jnp.sum(
                jnp.array(log_likelihood_poisson_vars), axis=(0, 1, 3)
            )
            poisson = jnp.sum(poisson_hosps) + jnp.sum(poisson_vars)

            ratio.iloc[i, 0] = jnp.sum(poisson_hosps) / poisson
            ratio.iloc[i, 1] = jnp.sum(nb_hosps) / nb
            i += 1

            hosps_diff = poisson_hosps - nb_hosps
            vars_diff = poisson_vars - nb_vars

            (
                inferer,
                runner,
                obs_hosps,
                obs_hosps_days,
                obs_sero_lmean,
                obs_sero_days,
                obs_var_prop,
                obs_var_days,
            ) = retrieve_inferer_obs(st, 0)

            fig, ax = plt.subplots()
            ax.plot(jnp.array(obs_var_days), vars_diff)
            ax.set_xlabel("Days")
            ax.set_ylabel("Log Difference")
            ax.set_title(f"{st} Hosps Log_Likelihood Difference: Poisson - NB")

            # Save the current figure to the PDF
            pdf.savefig(fig)
            plt.close(fig)  # Close the figure to free up memory
    ratio.index = states
    ratio.to_csv("output/ratio_log_likelihood.csv")
