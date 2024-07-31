# model_comparasion

import pandas as pd
import numpy as np
import arviz as az
import numpyro
import jax.numpy as jnp
from acc_metrics import mcmc_accuracy_measures

# both accuracy csv files-corresponding to each suffix- should have same US states, particles_per_chain and observed data, i.e same initial model day.


def compare_elpd_per_state(suffix1, suffix2):
    suffix1 = suffix1
    suffix2 = suffix2

    df1 = pd.read_csv(f"output/accuracy{suffix1}.csv")
    df2 = pd.read_csv(f"output/accuracy{suffix2}.csv")
    print(df1.columns[1:])
    if tuple(df1.columns) == tuple(df2.columns):
        print(df1.iloc[1, 1 : len(df1.columns)])
        print(df2.iloc[1, 1 : len(df1.columns)])
        l = []
        for k in range(1, len(df1.columns)):
            elpd1 = df1.iloc[0, k]
            elpd2 = df2.iloc[0, k]
            s1 = df1.iloc[1, k]
            s2 = df2.iloc[1, k]
            elpd1, elpd2, s1, s2 = float(elpd1), float(elpd2), float(s1), float(s2)
            z_score = np.abs(elpd1 - elpd2) / (np.sqrt(s1**2 + s2**2))
            if abs(z_score) > 2:
                l.append(
                    f"hospitalizations accuracy difference for state {df1.columns[k]} is significant"
                )
            else:
                l.append(
                    f"hospitalizations accuracy difference for state {df1.columns[k]} insignificant"
                )
        l = pd.Series(l, index=df1.columns[1:])
        df1 = pd.read_csv(f"output/accuracy{suffix1}.csv")
        df2 = pd.read_csv(f"output/accuracy{suffix2}.csv")
        l1 = []
        for k in range(1, len(df1.columns)):
            elpd1 = df1.iloc[7, k]
            elpd2 = df2.iloc[7, k]
            s1 = df1.iloc[8, k]
            s2 = df2.iloc[8, k]
            elpd1, elpd2, s1, s2 = float(elpd1), float(elpd2), float(s1), float(s2)
            z_score = np.abs(elpd2 - elpd1) / (np.sqrt(s1**2 + s2**2))
            if abs(z_score) > 2:
                l1.append(
                    f"variant prop accuracy difference for state {df2.columns[k]} is significant"
                )
            else:
                l1.append(
                    f"variant_prop accuracy difference for state {df2.columns[k]} insignificant"
                )
        l1 = pd.Series(l, index=df1.columns[1:])
        return l, l1
    else:
        print("states must be the same at same order")
        return []


def plot_epld_per_state_comparasion(
    state, particles_per_chain, initial_model_day, az_output, variant=False
):

    df_waic_0, waic_hosps_0, waic_vars_0 = mcmc_accuracy_measures(
        state=state,
        particles_per_chain=particles_per_chain,
        initial_model_day=initial_model_day,
        az_output=az_output[0],
        variant=variant,
    )

    df_waic_1, waic_hosps_1, waic_vars_1 = mcmc_accuracy_measures(
        state=state,
        particles_per_chain=particles_per_chain,
        initial_model_day=initial_model_day,
        az_output=az_output[1],
        variant=variant,
    )
    
    compare_dict_hosps={f'hospitalizations from the model whose azure output path is {az_output[0].split('/')[-1]}': waic_hosps_0,
                  f'hospitalizations from the model whose azure output path is {az_output[1].split('/')[-1]}': waic_hosps_1}
    
    compare_dict_vars={f'hospitalizations from the model whose azure output path is {az_output[0].split('/')[-1]}': waic_vars_0,
                  f'hospitalizations from the model whose azure output path is {az_output[1].split('/')[-1]}': waic_vars_1}
    
    return az.plot_elpd(compare_dict=compare_dict_hosps), az.plot_elpd(compare_dict=compare_dict_vars)


