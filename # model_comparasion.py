# %%
# model_comparasion
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, show
import matplotlib

matplotlib.use("TkAgg")
import pandas as pd
import numpy as np
import arviz as az
import numpyro
import jax.numpy as jnp
from acc_metrics import mcmc_accuracy_measures

az.style.use("arviz-doc")
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
            if abs(z_score) > 4:
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
            if z_score > 2:
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


# now we should plot the difference between the elpds. the az_output_path should be a list of two paths.
# should have fixed mcmc_accuracy_measures() variables.


def plot_elpd_per_state_comparasion(
    state, particles_per_chain, initial_model_day, az_output, ic, variant
):
    if variant == True:

        df_0, hosps_0, vars_0 = mcmc_accuracy_measures(
            state=state,
            particles_per_chain=particles_per_chain,
            initial_model_day=initial_model_day,
            az_output=az_output[0],
            ic=ic,
            variant=variant,
        )

        df_1, hosps_1, vars_1 = mcmc_accuracy_measures(
            state=state,
            particles_per_chain=particles_per_chain,
            initial_model_day=initial_model_day,
            az_output=az_output[1],
            ic=ic,
            variant=variant,
        )

        compare_dict_hosps = {
            f"hospitalizations from the model whose azure output path is {az_output[0]}": hosps_0,
            f"hospitalizations from the model whose azure output path is {az_output[1]}": hosps_1,
        }

        compare_dict_vars = {
            f"var_props from the model whose azure output is {az_output[0]}": vars_0,
            f"var_props from the model whose azure output is {az_output[1]}": vars_1,
        }

        return az.plot_elpd(
            compare_dict=compare_dict_hosps,
            threshold=2,
            ic=ic,
            xlabels=True,
        ), az.plot_elpd(
            compare_dict=compare_dict_vars,
            threshold=2,
            ic=ic,
            xlabels=True,
        )
    else:
        df_0, hosps_0 = mcmc_accuracy_measures(
            state=state,
            particles_per_chain=particles_per_chain,
            initial_model_day=initial_model_day,
            az_output=az_output[0],
            ic=ic,
            variant=variant,
        )

        df_1, hosps_1 = mcmc_accuracy_measures(
            state=state,
            particles_per_chain=particles_per_chain,
            initial_model_day=initial_model_day,
            az_output=az_output[1],
            ic=ic,
            variant=variant,
        )

        compare_dict_hosps = {
            f"hospitalizations {az_output[0]}": hosps_0,
            f"hospitalizations {az_output[1]}": hosps_1,
        }

        return az.plot_elpd(
            compare_dict=compare_dict_hosps,
            threshold=2,
            ic=ic,
            xlabels=False,
            show=False,
        )


def comparasion_per_state(
    state, particles_per_chain, initial_model_day, az_output, ic, variant
):
    if variant == True:

        df_0, hosps_0, vars_0 = mcmc_accuracy_measures(
            state=state,
            particles_per_chain=particles_per_chain,
            initial_model_day=initial_model_day,
            az_output=az_output[0],
            ic=ic,
            variant=variant,
        )

        df_1, hosps_1, vars_1 = mcmc_accuracy_measures(
            state=state,
            particles_per_chain=particles_per_chain,
            initial_model_day=initial_model_day,
            az_output=az_output[1],
            ic=ic,
            variant=variant,
        )

        compare_dict_hosps = {
            f"hospitalizations from the model whose azure output path is {az_output[0]}": hosps_0,
            f"hospitalizations from the model whose azure output path is {az_output[1]}": hosps_1,
        }

        compare_dict_vars = {
            f"var_props from the model whose azure output is {az_output[0]}": vars_0,
            f"var_props from the model whose azure output is {az_output[1]}": vars_1,
        }

        return az.compare(
            compare_dict=compare_dict_hosps,
        ), az.compare(
            compare_dict=compare_dict_vars,
        )
    else:
        df_0, hosps_0 = mcmc_accuracy_measures(
            state=state,
            particles_per_chain=particles_per_chain,
            initial_model_day=initial_model_day,
            az_output=az_output[0],
            ic=ic,
            variant=variant,
        )

        df_1, hosps_1 = mcmc_accuracy_measures(
            state=state,
            particles_per_chain=particles_per_chain,
            initial_model_day=initial_model_day,
            az_output=az_output[1],
            ic=ic,
            variant=variant,
        )

        compare_dict_hosps = {
            f"hospitalizations {az_output[0]}": hosps_0,
            f"hospitalizations {az_output[1]}": hosps_1,
        }

        return az.compare(
            compare_dict=compare_dict_hosps,
        )


def comparasion_plot(
    state, particles_per_chain, initial_model_day, az_output, ic, variant
):
    if variant == True:
        comparasion0, comparasion1 = comparasion_per_state(
            state, particles_per_chain, initial_model_day, az_output, ic, variant
        )
        return az.plot_compare(comparasion0), az.plot_compare(comparasion1)
    else:
        comparasion = comparasion_per_state(
            state, particles_per_chain, initial_model_day, az_output, ic, variant
        )
        return az.plot_compare(comparasion)


######################--------------------- Model_Comparasion.csv per state------------------------- ##########################################################

# df = comparasion_per_state(
#     state="CA",
#     particles_per_chain=80,
#     initial_model_day=560,
#     az_output=[
#         "/output/fifty_state_6strain_2204_2407/smh_6str_prelim_1/",
#         "/output/fifty_state_6strain_2204_2407/smh_6str_prelim_7/",
#     ],
#     ic="waic",
#     variant=False,
# )
# df.to_csv("output/model_comparasion_waic_CA.csv")


######### Plots individual ELPD difference per observed data. Useful to compare where observed data is scarse #########
ax0, ax1 = plot_elpd_per_state_comparasion(
    state="CA",
    particles_per_chain=90,
    initial_model_day=560,
    az_output=[
        "/output/fifty_state_6strain_2204_2407/smh_6str_prelim_1/",
        "/output/fifty_state_6strain_2204_2407/smh_6str_prelim_7/",
    ],
    ic="loo",
    variant=True,
)
fig1 = ax1.get_figure()
fig1.savefig("output/acc_fig_loo_var_prop_CA.png")


######### Plots the full model comparasion per state ###########


# ax0 = comparasion_plot(
#     state="CA",
#     particles_per_chain=80,
#     initial_model_day=560,
#     az_output=[
#         "/output/fifty_state_6strain_2204_2407/smh_6str_prelim_1/",
#         "/output/fifty_state_6strain_2204_2407/smh_6str_prelim_7/",
#     ],
#     ic="waic",
#     variant=False,
# )
# fig0 = ax0.get_figure()
# fig0.savefig("output/acc_fig_waic_hosps_CA.png")
