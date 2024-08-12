# %%
import copy
import datetime
import json
import multiprocessing as mp
import os
import random
import numpy as np

# from acc_metrics import *


import jax.numpy as jnp
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cycler import cycler
from matplotlib.backends.backend_pdf import PdfPages

from exp.fifty_state_6strain_2202_2407.inferer_smh_nb import SMHInferer
from exp.fifty_state_6strain_2202_2407.run_task import (
    rework_initial_state,
)
from mechanistic_model.covid_sero_initializer import CovidSeroInitializer
from mechanistic_model.mechanistic_runner import MechanisticRunner
from model_odes.seip_model import seip_ode2

plt.switch_backend("agg")
suffix = "_v2_6strain_nb"
az_output_path = "/output/fifty_state_2204_2407_6strain/ant-nb_higher_alpha"
pdf_filename = f"output/obs_vs_fitted{suffix}.pdf"
final_model_day = 890
initial_model_day = 0


# %%
def retrieve_inferer_obs(state, initial_model_day):
    state_config_path = os.path.join(az_output_path, state)
    print("Retrieving " + state + "\n")
    GLOBAL_CONFIG_PATH = os.path.join(state_config_path, "config_global_used.json")
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
    INFERER_CONFIG_PATH = os.path.join(state_config_path, "config_inferer_used.json")
    print()

    # sets up the initial conditions, initializer.get_initial_state() passed to runner
    initializer = CovidSeroInitializer(INITIALIZER_CONFIG_PATH, TEMP_GLOBAL_CONFIG_PATH)
    runner = MechanisticRunner(seip_ode2)
    initial_state = initializer.get_initial_state()
    initial_state = rework_initial_state(initial_state)
    inferer = SMHInferer(GLOBAL_CONFIG_PATH, INFERER_CONFIG_PATH, runner, initial_state)
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
        (hosp_data["day"] >= initial_model_day) & (hosp_data["day"] <= final_model_day)
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
        (sero_data["day"] >= initial_model_day) & (sero_data["day"] <= final_model_day)
    ].sort_values(by=["day", "age"], ascending=True, inplace=False)
    # transform data to logit scale
    sero_data["logit_rate"] = np.log(sero_data["rate"] / (100.0 - sero_data["rate"]))
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
        (var_data["day"] >= initial_model_day) & (var_data["day"] <= final_model_day)
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


def retrieve_post_samp(state):
    json_file = os.path.join(az_output_path, state, "checkpoint.json")
    post_samp = json.load(open(json_file, "r"))
    fitted_medians = {
        k: jnp.median(jnp.array(v), axis=(0, 1)) for k, v in post_samp.items()
    }

    return post_samp, fitted_medians


def replace_and_simulate(inferer, runner, fitted_medians):
    m = copy.deepcopy(inferer)
    m.config.INITIAL_INFECTIONS_SCALE = fitted_medians["INITIAL_INFECTIONS_SCALE"]
    m.config.INTRODUCTION_TIMES = [
        fitted_medians["INTRODUCTION_TIMES_0"],
        fitted_medians["INTRODUCTION_TIMES_1"],
        fitted_medians["INTRODUCTION_TIMES_2"],
        fitted_medians["INTRODUCTION_TIMES_3"],
        fitted_medians["INTRODUCTION_TIMES_4"],
    ]
    m.config.STRAIN_R0s = jnp.array(
        [
            fitted_medians["STRAIN_R0s_0"],
            fitted_medians["STRAIN_R0s_1"],
            fitted_medians["STRAIN_R0s_2"],
            fitted_medians["STRAIN_R0s_3"],
            fitted_medians["STRAIN_R0s_4"],
            fitted_medians["STRAIN_R0s_5"],
        ]
    )
    m.config.STRAIN_INTERACTIONS = jnp.array(
        [
            [
                fitted_medians["STRAIN_INTERACTIONS_0_0"],
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            [
                fitted_medians["STRAIN_INTERACTIONS_1_0"],
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            [
                fitted_medians["STRAIN_INTERACTIONS_2_0"],
                fitted_medians["STRAIN_INTERACTIONS_2_1"],
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            [
                0.5,
                fitted_medians["STRAIN_INTERACTIONS_3_1"],
                fitted_medians["STRAIN_INTERACTIONS_3_2"],
                1.0,
                1.0,
                1.0,
            ],
            [
                0.5,
                0.5,
                fitted_medians["STRAIN_INTERACTIONS_4_2"],
                fitted_medians["STRAIN_INTERACTIONS_4_3"],
                1.0,
                1.0,
            ],
            [
                0.5,
                0.5,
                0.5,
                fitted_medians["STRAIN_INTERACTIONS_5_3"],
                fitted_medians["STRAIN_INTERACTIONS_5_4"],
                1.0,
            ],
        ]
    )
    m.config.MIN_HOMOLOGOUS_IMMUNITY = fitted_medians["MIN_HOMOLOGOUS_IMMUNITY"]
    m.config.SEASONALITY_AMPLITUDE = fitted_medians["SEASONALITY_AMPLITUDE"]
    m.config.SEASONALITY_SHIFT = fitted_medians["SEASONALITY_SHIFT"]
    m.config.SEASONALITY_SECOND_WAVE = fitted_medians["SEASONALITY_SECOND_WAVE"]

    parameters = m.get_parameters()
    initial_state = m.scale_initial_infections(parameters["INITIAL_INFECTIONS_SCALE"])

    output = runner.run(
        initial_state,
        parameters,
        tf=final_model_day,
    )

    return output


def simulate_hospitalization(output, ihr, ihr_immune_mult, ihr_jn1_mult):
    model_incidence = jnp.diff(
        output.ys[3],
        axis=0,
    )

    model_incidence_no_exposures_non_jn1 = jnp.sum(
        model_incidence[:, :, 0, 0, :4], axis=-1
    )
    model_incidence_no_exposures_jn1 = jnp.sum(model_incidence[:, :, 0, 0, 4:], axis=-1)
    model_incidence_all_non_jn1 = jnp.sum(
        model_incidence[:, :, :, :, :4], axis=(2, 3, 4)
    )
    model_incidence_all_jn1 = jnp.sum(model_incidence[:, :, :, :, 4:], axis=(2, 3, 4))
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
        + model_incidence_w_exposures_jn1 * ihr * ihr_immune_mult * ihr_jn1_mult
    )

    return model_hosps


def plot_obsvfit(
    obs_hosps,
    sim_hosps_list,
    obs_hosps_days,
    obs_sero,
    sim_sero_list,
    obs_sero_days,
    obs_var_prop,
    sim_var_list,
    obs_var_days,
    inferer,
):
    dates = np.array(
        [
            inferer.config.INIT_DATE + datetime.timedelta(days=x)
            for x in range(final_model_day)
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
    axs[0].set_ylim([0.1, 500])
    # axs[0].legend(ncol=2)

    axs[1].xaxis.set_major_formatter(date_format)
    axs[1].set_prop_cycle(cycler(color=colors))
    for ss in sim_sero_list:
        axs[1].plot(dates, ss[1:], alpha=0.1)
    for s, c in zip(jnp.transpose(obs_sero), colors):
        axs[1].scatter(dates[obs_sero_days], s, color=c)
    axs[1].set_ylabel("Seroprevalence")
    axs[1].set_ylim([0, 1.1])

    colors2 = [
        "#1b9e77",
        "#d95f02",
        "#7570b3",
        "#e7298a",
        "#66a61e",
        "#e6ab02",
    ]
    axs[2].set_prop_cycle(cycler(color=colors2[:6]))
    axs[2].xaxis.set_major_formatter(date_format)
    for i, sv in enumerate(sim_var_list):
        if i == 0:
            axs[2].plot(
                dates[:final_model_day],
                sv[:, :final_model_day],
                alpha=0.1,
                label=inferer.config.STRAIN_IDX,
            )
        else:
            axs[2].plot(
                dates[:final_model_day],
                sv[:, :final_model_day],
                alpha=0.1,
            )
    for v, c in zip(jnp.transpose(obs_var_prop), colors2[:6]):
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
        obs_sero_lmean,
        obs_sero_days,
        obs_var_prop,
        obs_var_days,
    ) = retrieve_inferer_obs(state, initial_model_day)

    nsamp = len(samp["ihr_3"][0])
    nchain = len(samp["ihr_3"])
    ranindex = random.sample(list(range(nsamp)), 3)

    fitted_samples = [
        {k: v[c][r] for k, v in samp.items()} for r in ranindex for c in range(nchain)
    ]

    f = copy.deepcopy(fitted_medians)
    f["state"] = state
    #    median_df = pd.DataFrame(f, index=[state])

    obs_sero = 1 / (1 + jnp.exp(-obs_sero_lmean))
    sim_hosps_list = []
    sim_sero_list = []
    sim_var_list = []
    for f in fitted_samples:
        output = replace_and_simulate(inferer, runner, f)
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
        sim_hosps = simulate_hospitalization(output, ihr, ihr_immune_mult, ihr_jn1_mult)
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
        sim_vars = strain_incidence / jnp.sum(strain_incidence, axis=-1)[:, None]
        sim_var_list.append(sim_vars)

    # Visualization (Fitted vs Observed)
    fig = plot_obsvfit(
        obs_hosps,
        sim_hosps_list,
        obs_hosps_days,
        obs_sero,
        sim_sero_list,
        obs_sero_days,
        obs_var_prop,
        sim_var_list,
        obs_var_days,
        inferer,
    )

    return fig  # , median_df


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
    # "HI",
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
final_model_day = 890
initial_model_day = 0
states_omit = []
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
figs = pool.map(process_plot_state, [st for st in states])

# Now reset final_model_day, initial_model_day, if desired.

initial_model_day = 0

pdf_pages = PdfPages(pdf_filename)
for f in figs:
    pdf_pages.savefig(f)
    plt.close(f)
pdf_pages.close()

pool.close()
suffix = "v007"
# pd.concat(median_dfs).to_csv(f"output/medians{suffix}.csv", index=False)

# %%
