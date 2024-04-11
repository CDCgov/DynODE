# %%
import copy
import datetime
import types

import jax
import jax.config
import jax.numpy as jnp
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import pandas as pd
from cycler import cycler
from jax.random import PRNGKey
from numpyro.infer import MCMC, NUTS

from exp.fit_2year_1strain.inference import infer_model
from exp.fit_2year_1strain.initializer import EarlyCovidInitializer
from exp.fit_2year_1strain.utilities import (
    custom_beta_coef,
    deBoor,
    vaccination_rate,
)
from mechanistic_model.mechanistic_inferer import MechanisticInferer
from mechanistic_model.mechanistic_runner import MechanisticRunner
from model_odes.seip_model import seip_ode

numpyro.set_host_device_count(5)
jax.config.update("jax_enable_x64", True)

EXP_ROOT_PATH = "exp/fit_2year_1strain/"
GLOBAL_CONFIG_PATH = EXP_ROOT_PATH + "config_global.json"
INITIALIZER_CONFIG_PATH = EXP_ROOT_PATH + "config_initializer.json"
RUNNER_CONFIG_PATH = EXP_ROOT_PATH + "config_runner.json"
INFERER_CONFIG_PATH = EXP_ROOT_PATH + "config_inferer.json"

# Global
model_day = 567  # 2020-07-26 to 2022-02-11
knot_upper_bound = 570  # divisible by 30 but larger than model_day
ihr_knot_ub = 600  # divisible by 60 but larger than model_day
vax_start_day = 154  # 2020-12-13 but pushed back by 14 days because vax takes time to be effective


# %%
# Observations
## Incidence
obs_df = pd.read_csv("./data/hospitalization-data/hospital_200802_231231.csv")
obs_df["date"] = pd.to_datetime(obs_df["date"])
obs_reduced = (
    obs_df.groupby(["year", "week", "agegroup"])
    .agg({"date": "max", "new_admission_7": "mean"})
    .reset_index()
)  # Only taking last day per week
obs_reduced["day"] = (
    obs_reduced["date"] - pd.to_datetime("2020-07-26")
).dt.days - 7  # Assume infection to hospitalization delay by 7 days
obs_reduced = obs_reduced[obs_reduced["day"] <= model_day].reset_index()
obs_incidence = obs_reduced.groupby(["day"])["new_admission_7"].apply(np.array)
obs_days = jnp.array(
    list(obs_incidence.index)
)  # which days are the observations
obs_incidence = jnp.int64(jnp.array(obs_incidence.tolist()))

fig, ax = plt.subplots(1)
ax.plot(
    obs_days,
    np.asarray(obs_incidence),
    label=["0-17", "18-49", "50-64", "65+"],
)
fig.legend()
ax.set_title("Observed data")
plt.show()

# %%
# Seroprevalence
sero_df = pd.read_csv("data/serological-data/serology_all.csv")
sero_us = sero_df[sero_df.Site == "US"]
sero_us = sero_us[
    pd.to_datetime(sero_us.mid_date) >= pd.to_datetime("2020-08-10")
]  # Serology before 2020-08-10 is before the fitting horizon
obs_sero_lmean = sero_us.groupby(["mid_date"])["lmean"].apply(np.array)
sero_days = pd.to_datetime(obs_sero_lmean.index) - pd.to_datetime("2020-07-26")
sero_days = list(sero_days.days - 14)  # ASsume seroconversion takes two weeks
obs_sero_lmean = jnp.array(obs_sero_lmean.to_list())
obs_sero_lsd = sero_us.groupby(["mid_date"])["lsd"].apply(np.array)
obs_sero_lsd = jnp.array(obs_sero_lsd.to_list())

cond = [(d >= 9) and (d < model_day) for d in sero_days]
obs_sero_lmean = obs_sero_lmean[cond,]
obs_sero_lsd = obs_sero_lsd[cond,]
obs_sero_lsd = jnp.nanmean(obs_sero_lsd) * jnp.ones(obs_sero_lsd.shape)
# Make even fit to seroprevalences
sero_days = jnp.array(sero_days)[cond,]

# %%
# Initialize
initializer = EarlyCovidInitializer(
    INITIALIZER_CONFIG_PATH, GLOBAL_CONFIG_PATH, [0.102, 0.088, 0.065, 0.033]
)  # Serology informed how many already infected
initializer.load_init_infection_dist(
    age_split=jnp.array([0.40, 0.50, 0.05, 0.05])
)  # Arbitrarily set some good initial age split of infections
init_state = initializer.get_initial_state()


# %%
# Create runner and inferer, override beta_coef function
runner = MechanisticRunner(seip_ode)
inferer = MechanisticInferer(
    GLOBAL_CONFIG_PATH,
    INFERER_CONFIG_PATH,
    runner,
    init_state,
)

inferer.beta_coef = types.MethodType(custom_beta_coef, inferer)
inferer.vaccination_rate = types.MethodType(vaccination_rate, inferer)
inferer.config.VAX_TIMES = jnp.array([vax_start_day])
inferer.config.VAX_MODEL_DAYS_SHIFT = -vax_start_day

# %%
# Set default bspline knots and coefficients
k = list(np.arange(0.0, knot_upper_bound + 1, 30.0))
k = [0.0] * 3 + k + [knot_upper_bound] * 3
J = len(k) - 5

inferer.config.BSPLINE_KNOTS = jnp.array(k)
inferer.config.BSPLINE_COEFFS = jnp.array([1.0] * (J + 1))
# BSPLINE here code for beta multiplier

ihr_knots = list(np.arange(0.0, ihr_knot_ub + 1, 60.0))
ihr_knots = jnp.array([0.0] * 3 + ihr_knots + [ihr_knot_ub] * 3)
K = len(ihr_knots) - 5


# %%
# MCMC
init_values = {
    "INITIAL_INFECTIONS": 0.037,
    "ihr": jnp.array([0.0031, 0.0265, 0.0861, 0.3757]),
    "ihr_immune_mult": 0.32,
    "ihr_coef": jnp.array(
        [
            0.65838242,
            0.37618058,
            0.38432001,
            0.50770083,
            1.52434438,
            0.64213861,
            0.41244204,
            2.98790498,
            0.20722812,
            0.48371097,
            0.58330975,
            2.81403592,
        ]
    ),
    "sp_coef": jnp.array(
        [
            0.62072084,
            1.31050974,
            1.22409432,
            1.52925675,
            1.26402935,
            1.21021715,
            0.88448235,
            0.68687019,
            1.49998257,
            1.64300755,
            1.40311798,
            1.49860476,
            2.58045689,
            1.41982733,
            1.30641505,
            1.57285031,
            2.47649215,
            2.51117999,
            3.01710614,
            0.93344067,
            1.32575774,
        ]
    ),
}
nuts = NUTS(
    infer_model,
    dense_mass=True,
    max_tree_depth=7,
    # init_strategy=numpyro.infer.init_to_value(values=init_values),
    init_strategy=numpyro.infer.init_to_median(),
)
mcmc = MCMC(
    nuts,
    num_warmup=1000,
    num_samples=1000,
    thinning=1,
    num_chains=5,
    progress_bar=True,
)
with numpyro.validation_enabled(is_validate=True):
    mcmc.run(
        rng_key=PRNGKey(8811968),
        obs_incidence=obs_incidence,
        obs_days=obs_days,
        obs_sero_lmean=obs_sero_lmean,
        obs_sero_lsd=obs_sero_lsd / 20,
        # Increase weightage to seroprevalence vs hospitalization
        sero_days=sero_days,
        J=J,
        model_day=model_day,
        ihr_knots=ihr_knots,
        K=K,
        model=inferer,
        initializer=initializer,
    )
mcmc.print_summary()

# %%
# Extract samples and plot diagnostics
samp = mcmc.get_samples(group_by_chain=True)
fitted_medians_chain = {k: jnp.median(v, axis=1) for k, v in samp.items()}
fitted_medians = {k: jnp.median(v[:, -1], axis=0) for k, v in samp.items()}

fig, axs = plt.subplots(2, 2)
axs[0, 0].set_title("Coef[3]")
axs[0, 0].plot(np.transpose(samp["sp_coef"][:, :, 3]))
axs[0, 1].set_title("Coef[7]")
axs[0, 1].plot(np.transpose(samp["sp_coef"][:, :, 7]))
axs[1, 0].set_title("Coef[15]")
axs[1, 0].plot(np.transpose(samp["sp_coef"][:, :, 15]))
axs[1, 1].set_title("ihr_immune_mult")
axs[1, 1].plot(np.transpose(samp["ihr_immune_mult"]), label=range(5))
plt.legend()
plt.show()

# %%
# Run with fitted median parameters
m = copy.deepcopy(inferer)
m.config.BSPLINE_COEFFS = jnp.append(
    jnp.array([1.0]), fitted_medians["sp_coef"]
)
m.config.STRAIN_R0s = jnp.array([fitted_medians["r0"]])
initial_infections = (
    fitted_medians["INITIAL_INFECTIONS"] * 0.1 * initializer.config.POP_SIZE
)

output = runner.run(
    initializer.load_initial_state(initial_infections),
    m.get_parameters(),
    tf=model_day,
)

# %%
# Generate incidence of new run
ihr = fitted_medians["ihr"]
ihr_immune_mult = fitted_medians["ihr_immune_mult"]
ihr_coeffs = jnp.append(jnp.array([1.0]), fitted_medians["ihr_coef"])

ihr_mult_days = deBoor(
    jnp.searchsorted(ihr_knots, obs_days, "right") - 1,
    obs_days,
    ihr_knots,
    ihr_coeffs,
)

model_incidence = jnp.sum(output.ys[3], axis=4)
model_incidence_0 = jnp.diff(model_incidence[:, :, 0, 0], axis=0)
model_incidence_1 = jnp.sum(model_incidence, axis=(2, 3))
model_incidence_1 = jnp.diff(model_incidence_1, axis=0)
model_incidence_1 -= model_incidence_0

sim_incidence = (
    model_incidence_0 * ihr + model_incidence_1 * ihr * ihr_immune_mult
)[
    obs_days,
]
sim_incidence = sim_incidence * ihr_mult_days[:, None]

# %%
# Calculate sim seroprevalence and match with obs
never_infected = jnp.sum(output.ys[0][sero_days, :, 0, :, :], axis=(2, 3))
sim_seroprevalence = 1 - never_infected / m.config.POPULATION
sim_lseroprevalence = jnp.log(sim_seroprevalence / (1 - sim_seroprevalence))
obs_seroprevalence = 1 / (1 + jnp.exp(-obs_sero_lmean))

# %%
# Visualization (Fitted vs Observed)
dates = np.array(
    [m.config.INIT_DATE + datetime.timedelta(days=x) for x in range(model_day)]
)
date_format = mdates.DateFormatter("%b\n%y")
colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
fig, axs = plt.subplots(2, 1)
axs[0].xaxis.set_major_formatter(date_format)
axs[0].set_prop_cycle(cycler(color=colors))
axs[0].plot(
    dates[obs_days],
    np.log10(sim_incidence),
    label=["0-17", "18-49", "50-64", "65+"],
)
axs[0].plot(
    dates[obs_days],
    np.log10(obs_incidence),
    label=["0-17 (obs)", "18-49 (obs)", "50-64 (obs)", "65+ (obs)"],
    linestyle="dashed",
)
axs[0].set_title("Observed vs fitted")
axs[0].set_ylabel("Hospitalization (log10)")
axs[0].legend(ncol=2)

axs[1].xaxis.set_major_formatter(date_format)
axs[1].set_prop_cycle(cycler(color=colors))
sero_dates = [
    pd.to_datetime("2020-07-26") + pd.Timedelta(d, unit="day")
    for d in sero_days.tolist()
]
axs[1].plot(
    sero_dates,
    sim_seroprevalence,
    label=["0-17", "18-49", "50-64", "65+"],
)
for s, c in zip(jnp.transpose(obs_seroprevalence), colors):
    axs[1].scatter(sero_dates, s, color=c)

axs[1].set_ylabel("Seroprevalence")
fig.set_size_inches(8, 10)
fig.set_dpi(300)
plt.legend()
plt.show()

# %%
# Calculate time varying beta and IHR and visualize
beta_coef = [m.beta_coef(x) for x in range(model_day)]
fig, axs = plt.subplots(2, 1)
axs[0].xaxis.set_major_formatter(date_format)
axs[0].plot(dates[obs_days], ihr_mult_days)
axs[0].set_ylabel("IHR modifier")
axs[0].set_title("Time varying components")

axs[1].xaxis.set_major_formatter(date_format)
axs[1].plot(dates, beta_coef)
axs[1].set_ylabel("Beta multiplier")
fig.set_size_inches(8, 6)
fig.set_dpi(300)
plt.show()
