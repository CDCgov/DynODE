# %%
# model_comparasion
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, show
import matplotlib
from exp.fifty_state_6strain_2202_2407.postaz_process import retrieve_inferer_obs

matplotlib.use("TkAgg")
import pandas as pd
import numpy as np
import arviz as az
import numpyro
import jax.numpy as jnp
from acc_metrics import mcmc_accuracy_measures
import scipy.stats as stats

az.style.use("arviz-doc")

# now we should plot the difference between the elpds. the az_output_path should be a list of two paths.
# should have fixed mcmc_accuracy_measures() variables.


def plot_elpd_per_state_comparasion(
    state, particles_per_chain, initial_model_day, az_output, ic, variant
):
    (
        inferer,
        runner,
        obs_hosps,
        obs_hosps_days,
        obs_sero_lmean,
        obs_sero_days,
        obs_var_prop,
        obs_var_days,
    ) = retrieve_inferer_obs(state=state, initial_model_day=initial_model_day)
    age_group_colors = [f"C{i}" for i in range(obs_hosps.shape[-1])] * obs_hosps.shape[
        0
    ]
    var_prop_colors = [
        f"C{i}" for i in range(obs_var_prop.shape[-1])
    ] * obs_var_prop.shape[0]
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
            f"hosps from the model whose azure output path is {az_output[0]}": hosps_0,
            f"hosps from the model whose azure output path is {az_output[1]}": hosps_1,
        }

        compare_dict_vars = {
            f"var_props from the model whose azure output is {az_output[0]}": vars_0,
            f"var_props from the model whose azure output is {az_output[1]}": vars_1,
        }

        ax0, ax1 = az.plot_elpd(
            compare_dict=compare_dict_hosps,
            threshold=2,
            ic=ic,
            xlabels=True,
            color=age_group_colors,
            legend=True,
        ), az.plot_elpd(
            compare_dict=compare_dict_vars,
            threshold=2,
            ic=ic,
            xlabels=True,
            color=var_prop_colors,
            legend=True,
        )

        # Adjust legend

        # legend_labels = [f"Age Group {i}"for i in range(len(age_group_colors))]

        # handles0, _ = ax0.get_legend_handles_labels()

        # ax0.legend(handles0=handles[:len(age_group_colors)], labels=legend_labels, title="Age Groups")

        # Adjust x-axis labels to show only the number of weeks

        return ax0, ax1
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

        ax = az.plot_elpd(
            compare_dict=compare_dict_hosps,
            threshold=2,
            ic=ic,
            xlabels=True,
            show=False,
            color=age_group_colors,
            legend=True,
        )

        return ax


# both models should have same US state, particles_per_chain and observed data.


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

        df0 = az.compare(
            compare_dict=compare_dict_hosps,
        )
        p = df0["elpd_diff"] / df0["dse"]
        p_value = 2 * (1 - stats.norm.cdf(abs(p)))

        df0["p_value"] = p_value

        df1 = az.compare(
            compare_dict=compare_dict_vars,
        )
        p = df1["elpd_diff"] / df1["dse"]
        p_value = 2 * (1 - stats.norm.cdf(abs(p)))

        df1["p_value"] = p_value

        return df0, df1
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
        df = az.compare(
            compare_dict=compare_dict_hosps,
        )
        p = df["elpd_diff"] / df["dse"]
        p_value = 2 * (1 - stats.norm.cdf(abs(p)))

        df["p_value"] = p_value
        return df


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


##################################--------------------- Model_Comparasion.csv per state------------------------- ##########################################################

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
    particles_per_chain=250,
    initial_model_day=690,
    az_output=[
        "/output/fifty_state_6strain_2204_2407/smh_6str_prelim_1/",
        "/output/fifty_state_6strain_2204_2407/smh_6str_prelim_7/",
    ],
    ic="loo",
    variant=True,
)
fig0 = ax0.get_figure()
fig0.savefig("output/testing_loo_hosp_CA.png")
fig1 = ax1.get_figure()
fig1.savefig("output/testing_loo_var_prop_CA.png")


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
