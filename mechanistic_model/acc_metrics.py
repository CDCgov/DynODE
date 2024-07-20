from exp.fifty_state_6strain_2202_2407.postaz_process import (retrieve_inferer_obs, retrieve_post_samp, replace_and_simulate, simulate_hospitalization )
import arviz as az
from sklearn.metrics import mean_squared_error
import numpy as np
import jax.numpy as jnp
import jax
from jax import vmap, jit
from functools import partial
# import copy
# import datetime
# import json
import multiprocessing as mp
import os
import random
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# from matplotlib.backends.backend_pdf import PdfPages

# plt.switch_backend("agg") 
# suffix = "6strains_v0"
# az_output_path = "/output/fifty_state_6strain_2202_2407/smh_6str_prelim_3/"
# pdf_filename = f"output/acc_{suffix}.pdf"
# final_model_day = 890
# initial_model_day = 0

def hosp_var_posterior(samp, inferer, runner, particles_per_chain, final_model_day, initial_model_day):
    nsamp = len(samp["ihr_3"][0])  # Number of samples per chain
    nchain = len(samp["ihr_3"])  # Number of chains
    # each element of the list all_samples is a dictionary of unique values for each parameter from the ODE.
    # the list contains-if you consider a "vertical line" across all dictionaries- the 4 particles, as we have 4 chains for each parameter.
    # Randomly select a subset of posterior samples for simulation
    ranindex = random.sample(
        list(range(nsamp)), particles_per_chain
    )  # Randomly select 25 samples per chain
    all_samples = [
        {k: v[c][r] for k, v in samp.items()} for r in ranindex for c in range(nchain)
    ]
    pred_hosps_list = []
    pred_var_list = []
    
    def process_per_sample(inferer, runner, f):
        output = replace_and_simulate(inferer, runner, f, final_model_day, initial_model_day)
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
        pred_hosps = simulate_hospitalization(output, ihr, ihr_immune_mult, ihr_jn1_mult)
 
        strain_incidence = jnp.sum(
            output.ys[inferer.config.COMPARTMENT_IDX.C],
            axis=(
                inferer.config.I_AXIS_IDX.age + 1,
                inferer.config.I_AXIS_IDX.hist + 1,
                inferer.config.I_AXIS_IDX.vax + 1,
            ),
        )
        strain_incidence = jnp.diff(strain_incidence, axis=0)
        pred_vars = strain_incidence / jnp.sum(strain_incidence, axis=-1)
 
        return pred_hosps, pred_vars
 
    # Vectorize the function
    vectorized_process_sample = vmap(partial(process_per_sample, inferer, runner))
 
    # JIT compile the vectorized function
    jit_vectorized_process_sample = jit(vectorized_process_sample)
 
    # Convert the list of dictionaries to a dictionary of lists (if not already)
    all_samples_dict = {key: jnp.array([f[key] for f in all_samples]) for key in all_samples[0]}
 
    # Apply the JIT-compiled and vectorized function
    pred_hosps_array, pred_vars_array = jit_vectorized_process_sample(all_samples_dict)
 
    # Convert the arrays to lists if needed
    pred_hosps_list = list(pred_hosps_array)
    pred_var_list = list(pred_vars_array)
    
    return pred_hosps_list, pred_var_list

#ESTE ULTIMO TEM QUE TER APENAS state COMO PARAMETRO
def mcmc_accuracy_measures(state, final_model_day, initial_model_day, samp, particles_per_chain):
    
    (
        inferer,
        runner,
        obs_hosps,
        obs_hosps_days,
        obs_sero_lmean,
        obs_sero_days,
        obs_var_prop,
        obs_var_days,
    )=retrieve_inferer_obs(state,final_model_day,initial_model_day)
    pred_hosps_list, pred_var_list=hosp_var_posterior(samp, inferer, runner, particles_per_chain, final_model_day, initial_model_day)
    trace_hosps = az.from_dict(posterior_predictive={"hospitalizations": pred_hosps_list},
                               observed_data={"hospitalizations": obs_hosps})
    waic_hosps = az.waic(trace_hosps)
    print("WAIC for Hospitalizations:", waic_hosps)
    # WAIC for variant proportions
    waic_strain_results={}
    for strain in range(pred_var_list[0].shape[-1]):
        trace_vars_strain=az.from_dict(posterior_predictive={"variant proportions":[p[:, strain] for p in pred_var_list]},
                                       observed_data={"variant_proportions:" }: obs_var_prop[:, strain])
        waic_strain_results[f"WAIC for strain {strain}"]=az.waic(trace_vars_strain)
    # for strain, waic in waic_strain_results.items():
    #     print(f"strain {strain} has WAIC: {waic}")
    rmse={}
    rmse_hosp=np.sqrt(mean_squared_error(jnp.median(pred_hosps_list),obs_hosps))
    for strain in range(pred_var_list[0].shape[-1]):
        rmse[f"RMSE strain {strain}"]=np.sqrt(mean_squared_error(jnp.median(pred_var_list[:,:,strain]),obs_hosps))
    df_rmse=pd.DataFrame(rmse)
    df_rmse['RMSE Hospitalizations']=rmse_hosp
    
    
    df_waic=pd.DataFrame(waic_strain_results)
    df_waic['WAIC for hospitalizations']=waic_hosps
    df_total=pd.concat([df_waic,df_rmse])
    
    return df_total
