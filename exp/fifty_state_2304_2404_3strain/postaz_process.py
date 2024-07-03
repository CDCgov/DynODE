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

from exp.fifty_state_2304_2404_3strain.epoch_two_initializer import (
    smh_initializer_epoch_two,
)
from exp.fifty_state_2304_2404_3strain.inferer_smh import SMHInferer
from mechanistic_model.mechanistic_runner import MechanisticRunner
from model_odes.seip_model import seip_ode

plt.switch_backend("agg")
model_day = 450
suffix = "_v15b6"
az_output_path = (
    "/output/fifty_state_2304_2404_3strain/smh_epoch_2_tryout6_240504"
)
pdf_filename = f"output/obs_vs_fitted{suffix}.pdf"
pdf_pages = PdfPages(pdf_filename)


# %%
def retrieve_inferer_obs(state):
    state_config_path = os.path.join(az_output_path, state)
    print("Retrieving " + state + "\n")
    GLOBAL_CONFIG_PATH = os.path.join(
        state_config_path, "config_global_used.json"
    )
    INITIALIZER_CONFIG_PATH = os.path.join(
        state_config_path, "config_initializer_used.json"
    )
    INFERER_CONFIG_PATH = os.path.join(
        state_config_path, "config_inferer_used.json"
    )

    # sets up the initial conditions, initializer.get_initial_state() passed to runner
    initializer = smh_initializer_epoch_two(
        INITIALIZER_CONFIG_PATH, GLOBAL_CONFIG_PATH
    )
    runner = MechanisticRunner(seip_ode)
    initial_state = initializer.get_initial_state()
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

    var_data_filename = "%s_strain_prop.csv" % (
        initializer.config.REGIONS[0].replace(" ", "_")
    )
    var_data_path = os.path.join(inferer.config.VAR_PATH, var_data_filename)
    # currently working up to third strain which is XBB1
    var_data = pd.read_csv(var_data_path)
    var_data = var_data[var_data["strain"] >= 2]
    var_data["date"] = pd.to_datetime(var_data["date"])
    var_data["day"] = (
        var_data["date"] - pd.to_datetime(inferer.config.INIT_DATE)
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
        obs_var_prop,
        obs_var_days,
    )


def retrieve_post_samp(state):
    json_file = os.path.join(az_output_path, state, "checkpoint.json")
    post_samp = json.load(open(json_file, "r"))
    post_samp.pop("final_timestep_s", None)
    post_samp.pop("final_timestep_e", None)
    post_samp.pop("final_timestep_i", None)
    post_samp.pop("final_timestep_c", None)
    fitted_medians = {
        k: jnp.median(jnp.array(v), axis=(0, 1)) for k, v in post_samp.items()
    }

    return post_samp, fitted_medians


def replace_and_simulate(inferer, runner, fitted_medians):
    m = copy.deepcopy(inferer)
    m.config.INTRODUCTION_TIMES = [
        fitted_medians["INTRODUCTION_TIMES_0"],
        fitted_medians["INTRODUCTION_TIMES_1"],
    ]
    m.config.INITIAL_INFECTIONS_SCALE = fitted_medians[
        "INITIAL_INFECTIONS_SCALE"
    ]
    m.config.STRAIN_R0s = jnp.array(
        [
            fitted_medians["STRAIN_R0s_0"],
            fitted_medians["STRAIN_R0s_1"],
            fitted_medians["STRAIN_R0s_2"],
        ]
    )
    m.config.STRAIN_INTERACTIONS = jnp.array(
        [
            [fitted_medians["STRAIN_INTERACTIONS_0"], 1.0, 1.0],
            [fitted_medians["STRAIN_INTERACTIONS_3"], 1.0, 1.0],
            [
                fitted_medians["STRAIN_INTERACTIONS_6"],
                fitted_medians["STRAIN_INTERACTIONS_7"],
                1.0,
            ],
        ]
    )
    m.config.MIN_HOMOLOGOUS_IMMUNITY = fitted_medians[
        "MIN_HOMOLOGOUS_IMMUNITY"
    ]
    m.config.SEASONALITY_AMPLITUDE = fitted_medians["SEASONALITY_AMPLITUDE"]
    m.config.SEASONALITY_SHIFT = fitted_medians["SEASONALITY_SHIFT"]

    parameters = m.get_parameters()
    initial_state = m.scale_initial_infections(
        parameters["INITIAL_INFECTIONS_SCALE"]
    )
    output = runner.run(
        initial_state,
        parameters,
        tf=model_day,
    )

    return output


def simulate_hospitalization(output, ihr, ihr_immune_mult, ihr_jn1_mult):
    model_incidence = jnp.diff(output.ys[3], axis=0)

    model_incidence_no_exposures_non_jn1 = jnp.sum(
        model_incidence[:, :, 0, 0, :2], axis=-1
    )
    model_incidence_no_exposures_jn1 = model_incidence[:, :, 0, 0, 2]
    model_incidence_all_non_jn1 = jnp.sum(
        model_incidence[:, :, :, :, :2], axis=(2, 3, 4)
    )
    model_incidence_all_jn1 = jnp.sum(
        model_incidence[:, :, :, :, 2], axis=(2, 3)
    )
    model_incidence_w_exposures_non_jn1 = (
        model_incidence_all_non_jn1 - model_incidence_no_exposures_non_jn1
    )
    model_incidence_w_exposures_jn1 = (
        model_incidence_all_jn1 - model_incidence_no_exposures_jn1
    )

    model_hosps = (
        model_incidence_no_exposures_non_jn1 * ihr
        + model_incidence_no_exposures_jn1 * ihr * ihr_jn1_mult
        + model_incidence_w_exposures_non_jn1 * ihr * ihr_immune_mult
        + model_incidence_w_exposures_jn1
        * ihr
        * ihr_immune_mult
        * ihr_jn1_mult
    )
    return model_hosps


def plot_obsvfit(
    obs_hosps,
    sim_hosps_list,
    obs_hosps_days,
    sim_sero_list,
    obs_var_prop,
    sim_var_list,
    obs_var_days,
    inferer,
):
    dates = np.array(
        [
            inferer.config.INIT_DATE + datetime.timedelta(days=x)
            for x in range(model_day)
        ]
    )
    date_format = mdates.DateFormatter("%b\n%y")
    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
    fig, axs = plt.subplots(3, 1)
    axs[0].xaxis.set_major_formatter(date_format)
    axs[0].set_yscale("log")
    axs[0].set_prop_cycle(cycler(color=colors))
    for i, sh in enumerate(sim_hosps_list):
        if i == 0:
            axs[0].plot(
                dates,
                sh,
                label=["0-17", "18-49", "50-64", "65+"],
                alpha=0.1,
            )
        else:
            axs[0].plot(dates, sh, alpha=0.1)

    axs[0].plot(dates[obs_hosps_days], obs_hosps, linestyle=":")
    # for h, c in zip(obs_hosps.T, colors):
    #     axs[0].scatter(dates[obs_hosps_days], np.log10(h), s=4, color=c)
    axs[0].set_title(inferer.config.REGIONS[0] + ": Observed vs fitted")
    axs[0].set_ylabel("Hospitalization")
    # axs[0].legend(ncol=2)

    axs[1].xaxis.set_major_formatter(date_format)
    axs[1].set_prop_cycle(cycler(color=colors))
    for ss in sim_sero_list:
        axs[1].plot(dates, ss[1:], alpha=0.1)
    axs[1].set_ylabel("Seroprevalence")

    colors2 = [
        "#7fc97f",
        "#beaed4",
        "#fdc086",
        "#ffff99",
        "#386cb0",
        "#f0027f",
    ]
    axs[2].set_prop_cycle(cycler(color=colors2[:3]))
    axs[2].xaxis.set_major_formatter(date_format)
    for i, sv in enumerate(sim_var_list):
        if i == 0:
            axs[2].plot(
                dates[:model_day],
                sv[:, :model_day],
                alpha=0.1,
                label=inferer.config.STRAIN_IDX,
            )
        else:
            axs[2].plot(
                dates[:model_day],
                sv[:, :model_day],
                alpha=0.1,
            )
    for v, c in zip(jnp.transpose(obs_var_prop), colors2[:3]):
        axs[2].scatter(dates[obs_var_days], v, color=c, s=7)
    axs[2].set_ylabel("Variant proportion")

    fig.set_size_inches(8, 10)
    fig.set_dpi(300)
    leg = fig.legend()
    for lh in leg.legend_handles:
        lh.set_alpha(1)
    return fig


def process_plot_state(state):
    samp, fitted_medians = retrieve_post_samp(state)
    (
        inferer,
        runner,
        obs_hosps,
        obs_hosps_days,
        obs_var_prop,
        obs_var_days,
    ) = retrieve_inferer_obs(state)

    nsamp = len(samp["ihr_0"][0])
    nchain = len(samp["ihr_0"])
    ranindex = random.sample(list(range(nsamp)), 3)

    fitted_samples = [
        {k: v[c][r] for k, v in samp.items()}
        for r in ranindex
        for c in range(nchain)
    ]

    f = copy.deepcopy(fitted_medians)
    f["state"] = state
    median_df = pd.DataFrame(f, index=[state])

    sim_hosps_list = []
    sim_sero_list = []
    sim_var_list = []
    for f in fitted_samples:
        output = replace_and_simulate(inferer, runner, f)
        ihr = jnp.array(
            [
                f["ihr_0"],
                f["ihr_1"],
                f["ihr_2"],
                f["ihr_3"],
            ]
        )
        ihr_immune_mult = f["ihr_immune_mult"]
        ihr_jn1_mult = f["ihr_jn1_mult"]
        sim_hosps = simulate_hospitalization(
            output, ihr, ihr_immune_mult, ihr_jn1_mult
        )
        # sim_hosps = sim_hosps[obs_hosps_days,]
        sim_hosps_list.append(sim_hosps)

        never_infected = jnp.sum(output.ys[0][:, :, 0, :, :], axis=(2, 3))
        sim_sero = 1 - never_infected / inferer.config.POPULATION
        sim_sero_list.append(sim_sero)

        strain_incidence = jnp.sum(
            output.ys[inferer.config.COMPARTMENT_IDX.C],
            axis=(
                inferer.config.I_AXIS_IDX.age + 1,
                inferer.config.I_AXIS_IDX.hist + 1,
                inferer.config.I_AXIS_IDX.vax + 1,
            ),
        )
        strain_incidence = jnp.diff(strain_incidence, axis=0)
        sim_vars = (
            strain_incidence / jnp.sum(strain_incidence, axis=-1)[:, None]
        )
        sim_var_list.append(sim_vars)

    # Visualization (Fitted vs Observed)
    fig = plot_obsvfit(
        obs_hosps,
        sim_hosps_list,
        obs_hosps_days,
        sim_sero_list,
        obs_var_prop,
        sim_var_list,
        obs_var_days,
        inferer,
    )

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
    json_file = os.path.join(az_output_path, st, "checkpoint.json")
    if not os.path.exists(json_file):
        states_omit.append(st)
states = list(set(states).difference(set(states_omit)))
states.sort()
# states = ["CT"]
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

# %%
