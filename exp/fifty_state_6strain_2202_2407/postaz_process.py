# %%
import copy
import datetime
import json
import multiprocessing as mp
import os
import random

import jax.numpy as jnp
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cycler import cycler
from matplotlib.backends.backend_pdf import PdfPages

from exp.fifty_state_6strain_2202_2407.inferer_smh import SMHInferer
from exp.fifty_state_6strain_2202_2407.run_task import (
    rework_initial_state,
)
from mechanistic_model.covid_sero_initializer import CovidSeroInitializer
from mechanistic_model.mechanistic_runner import MechanisticRunner
from model_odes.seip_model import seip_ode2

plt.switch_backend("agg")
model_day = 890
suffix = "_v4_6strain"
az_output_path = "/output/fifty_state_6strain_2202_2407/smh_6str_prelim_6/"
pdf_filename = f"output/obs_vs_fitted{suffix}.pdf"


# %%
def retrieve_inferer_obs(state):
    state_config_path = os.path.join(az_output_path, state)
    print("Retrieving " + state + "\n")
    GLOBAL_CONFIG_PATH = os.path.join(
        state_config_path, "config_global_used.json"
    )
    TEMP_GLOBAL_CONFIG_PATH = os.path.join(
        state_config_path, "temp_config_global_template.json"
    )
    global_js = json.load(open(GLOBAL_CONFIG_PATH))
    global_js["NUM_STRAINS"] = 3
    global_js["NUM_WANING_COMPARTMENTS"] = 5
    global_js["WANING_TIMES"] = [70, 70, 70, 129, 0]
    json.dump(global_js, open(TEMP_GLOBAL_CONFIG_PATH, "w"))
    INITIALIZER_CONFIG_PATH = os.path.join(
        state_config_path, "config_initializer_used.json"
    )
    INFERER_CONFIG_PATH = os.path.join(
        state_config_path, "config_inferer_used.json"
    )

    # sets up the initial conditions, initializer.get_initial_state() passed to runner
    initializer = CovidSeroInitializer(
        INITIALIZER_CONFIG_PATH, TEMP_GLOBAL_CONFIG_PATH
    )
    runner = MechanisticRunner(seip_ode2)
    initial_state = initializer.get_initial_state()
    initial_state = rework_initial_state(initial_state)
    inferer = SMHInferer(
        GLOBAL_CONFIG_PATH, INFERER_CONFIG_PATH, runner, initial_state
    )
    # observed data
    hosp_data_filename = "%s_hospitalization.csv" % (
        initializer.config.REGIONS[0].replace(" ", "_")
    )
    hosp_data_path = os.path.join(inferer.config.HOSP_PATH, hosp_data_filename)
    hosp_data = pd.read_csv(hosp_data_path)
    hosp_data["date"] = pd.to_datetime(hosp_data["date"])
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

    var_data_filename = "%s_strain_prop.csv" % (
        initializer.config.REGIONS[0].replace(" ", "_")
    )
    var_data_path = os.path.join(inferer.config.VAR_PATH, var_data_filename)
    # currently working up to third strain which is XBB1
    var_data = pd.read_csv(var_data_path)
    var_data = var_data[var_data["strain"] < 6]
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
    obs_var_prop = obs_var_prop / jnp.sum(obs_var_prop, axis=1)[:, None]

    return (
        inferer,
        runner,
        obs_hosps,
        obs_hosps_days,
        obs_sero_lmean,
        obs_sero_days,
        obs_var_prop,
        obs_var_days,
    )


def retrieve_fitted_medians(state):
    json_file = os.path.join(az_output_path, state, "checkpoint.json")
    post_samp = json.load(open(json_file, "r"))
    fitted_medians = {
        k: jnp.median(jnp.array(v), axis=(0, 1)) for k, v in post_samp.items()
    }

    return fitted_medians


def retrieve_timeline(state):
    csv_file = os.path.join(
        az_output_path, state, "azure_visualizer_timeline.csv"
    )
    timeline = pd.read_csv(csv_file)
    timeline["date"] = pd.to_datetime(timeline["date"])

    return timeline


def process_plot_state(state):
    timeline = retrieve_timeline(state)
    fitted_medians = retrieve_fitted_medians(state)
    fitted_medians["state"] = state
    median_df = pd.DataFrame(fitted_medians, index=[state])
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

    obs_sero = 1 / (1 + np.exp(-obs_sero_lmean))
    chain_particles = timeline["chain_particle"].unique()
    date_format = mdates.DateFormatter("%b\n%y")
    colors_age = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
    colors_strain = [
        "#1b9e77",
        "#d95f02",
        "#7570b3",
        "#e7298a",
        "#66a61e",
        "#e6ab02",
    ]
    fig, axs = plt.subplots(3, 1)
    # Configs
    axs[0].xaxis.set_major_formatter(date_format)
    axs[0].set_prop_cycle(cycler(color=colors_age))
    axs[0].set_title(state + ": Observed vs fitted")
    axs[0].set_ylim([0.01, 500])
    axs[0].set_yscale("log")
    axs[0].set_ylabel("Hospitalization")

    axs[1].xaxis.set_major_formatter(date_format)
    axs[1].set_prop_cycle(cycler(color=colors_age))
    axs[1].set_ylabel("Seroprevalence")
    axs[1].set_ylim([0, 1.1])

    axs[2].set_prop_cycle(cycler(color=colors_strain[:6]))
    axs[2].xaxis.set_major_formatter(date_format)
    axs[2].set_ylabel("Variant proportion")
    # Simulated
    for i, cp in enumerate(chain_particles):
        sub_df = timeline[timeline["chain_particle"] == cp]
        sim_hosp = np.array(
            sub_df[
                [
                    "pred_hosp_0_17",
                    "pred_hosp_18_49",
                    "pred_hosp_50_64",
                    "pred_hosp_65+",
                ]
            ]
        )
        sim_sero = np.array(
            sub_df[["sero_0_17", "sero_18_49", "sero_50_64", "sero_65+"]]
        )
        sp_columns = [x for x in sub_df.columns if "strain_proportion" in x]
        sim_var_prop = np.array(sub_df[sp_columns])
        if i == 0:
            dates = np.array(sub_df["date"])
            axs[0].plot(
                dates,
                sim_hosp,
                label=["0-17", "18-49", "50-64", "65+"],
                alpha=0.1,
            )
            axs[2].plot(
                dates,
                sim_var_prop,
                alpha=0.1,
                label=inferer.config.STRAIN_IDX,
            )
        else:
            axs[0].plot(dates, sim_hosp, alpha=0.1)
            axs[2].plot(
                dates,
                sim_var_prop,
                alpha=0.1,
            )

        axs[1].plot(dates, sim_sero, alpha=0.1)

    # Observed
    axs[0].plot(dates[obs_hosps_days], obs_hosps, linestyle=":")
    for s, c in zip(jnp.transpose(obs_sero), colors_age):
        axs[1].scatter(dates[obs_sero_days], s, color=c)
    for v, c in zip(jnp.transpose(obs_var_prop), colors_strain[:6]):
        axs[2].scatter(dates[obs_var_days], v, color=c, s=7)

    fig.set_size_inches(8, 10)
    fig.set_dpi(300)
    leg = fig.legend(loc=7)
    for lh in leg.legend_handles:
        lh.set_alpha(1)
    return fig, median_df


# %%
states = [
    "AL",
    "AK",
    "AZ",
    "AR",
    "CA",
    "CO",
    "CT",
    "DE",
    "FL",
    "GA",
    "HI",
    "ID",
    "IL",
    "IN",
    "IA",
    "KS",
    "KY",
    "LA",
    "ME",
    "MD",
    "MA",
    "MI",
    "MN",
    "MS",
    "MO",
    "MT",
    "NE",
    "NV",
    "NH",
    "NJ",
    "NM",
    "NY",
    "NC",
    "ND",
    "OH",
    "OK",
    "OR",
    "PA",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "UT",
    "VT",
    "VA",
    "WA",
    "WV",
    "WI",
    "WY",
]
states_omit = []
for st in states:
    csv_file = os.path.join(
        az_output_path, st, "azure_visualizer_timeline.csv"
    )
    if not os.path.exists(csv_file):
        states_omit.append(st)
states = list(set(states).difference(set(states_omit)))
states.sort()
# states = ["CA", "IN", "IL", "WA"]
# states = ["US"] + states
print(states)


# %%
pool = mp.Pool(5)
figs, median_dfs = zip(*pool.map(process_plot_state, [st for st in states]))

pdf_pages = PdfPages(pdf_filename)
for f in figs:
    pdf_pages.savefig(f)
    plt.close(f)
pdf_pages.close()

pool.close()
pd.concat(median_dfs).to_csv(f"output/medians{suffix}.csv", index=False)
