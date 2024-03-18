# %%
import copy
import datetime
import os
import types

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

os.chdir("/home/bentoh/projects/scenarios-2/")
from exp.fit_3year_1strain.inference import infer_model
from exp.fit_3year_1strain.initializer import EarlyCovidInitializer
from exp.fit_3year_1strain.utilities import custom_beta_coef, vaccination_rate
from mechanistic_model.mechanistic_inferer import MechanisticInferer
from mechanistic_model.mechanistic_runner import MechanisticRunner
from model_odes.seip_model import seip_ode

numpyro.set_host_device_count(5)
jax.config.update("jax_enable_x64", True)

# Paths
EXP_ROOT_PATH = "exp/fit_3year_1strain/"
GLOBAL_CONFIG_PATH = EXP_ROOT_PATH + "config_global.json"
INITIALIZER_CONFIG_PATH = EXP_ROOT_PATH + "config_initializer.json"
RUNNER_CONFIG_PATH = EXP_ROOT_PATH + "config_runner.json"
INFERER_CONFIG_PATH = EXP_ROOT_PATH + "config_inferer.json"

# Global
obs_days = 250
knot_upper_bound = 270
vax_start_day = 140

# %%
# Observations
## Incidence
obs_df = pd.read_csv("./data/hospitalization-data/hospital_200802_231231.csv")
obs_incidence = obs_df.groupby(["date"])["new_admission_7"].apply(np.array)
obs_incidence = jnp.array(obs_incidence.tolist())
obs_incidence = obs_incidence[:obs_days]

fig, ax = plt.subplots(1)
ax.plot(np.asarray(obs_incidence), label=["0-17", "18-49", "50-64", "65+"])
fig.legend()
ax.set_title("Observed data")
plt.show()

# %%
# Seroprevalence
sero_df = pd.read_csv("data/serological-data/serology_all.csv")
sero_us = sero_df[sero_df.Site == "US"]
sero_us = sero_us[
    pd.to_datetime(sero_us.mid_date) <= pd.to_datetime("2022-02-11")
]
obs_sero_lmean = sero_us.groupby(["mid_date"])["logit_mean"].apply(np.array)
sero_days = pd.to_datetime(obs_sero_lmean.index) - pd.to_datetime("2020-07-26")
sero_days = list(sero_days.days - 21)
obs_sero_lmean = jnp.array(obs_sero_lmean.to_list())
obs_sero_lsd = sero_us.groupby(["mid_date"])["logit_sd"].apply(np.array)
obs_sero_lsd = jnp.array(obs_sero_lsd.to_list())

cond = [(d >= 10) and (d < obs_days) for d in sero_days]
obs_sero_lmean = obs_sero_lmean[cond,]
obs_sero_lsd = obs_sero_lsd[cond,]
sero_days = jnp.array(sero_days)[cond,]

# %%
# Initialize
initializer = EarlyCovidInitializer(
    INITIALIZER_CONFIG_PATH, GLOBAL_CONFIG_PATH, [0.079, 0.068, 0.05, 0.025]
)
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
# inferer.zero_vaccination_rate = zero_vaccination_rate
# inferer.spline_vaccination_rate = spline_vaccination_rate
inferer.config.VAX_TIMES = jnp.array([vax_start_day])
inferer.config.VAX_MODEL_DAYS_SHIFT = -vax_start_day

# %%
# Set default bspline knots and coefficients
k = list(np.arange(0.0, knot_upper_bound + 1, 30.0))
k = [0.0] * 3 + k + [knot_upper_bound] * 3
J = len(k) - 5

inferer.config.BSPLINE_KNOTS = jnp.array(k)
inferer.config.BSPLINE_COEFFS = jnp.array([1.0] * (J + 1))

# %%
# MCMC
nuts = NUTS(
    infer_model,
    dense_mass=True,
    max_tree_depth=6,
    init_strategy=numpyro.infer.init_to_median(),
    target_accept_prob=0.80,
    # find_heuristic_step_size=True,
)
mcmc = MCMC(
    nuts,
    num_warmup=1000,
    num_samples=1000,
    num_chains=5,
    progress_bar=True,
)
mcmc.run(
    rng_key=PRNGKey(8811968),
    obs_incidence=obs_incidence,
    obs_sero_lmean=obs_sero_lmean,
    obs_sero_lsd=obs_sero_lsd / 10,
    sero_days=sero_days,
    J=J,
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
axs[1, 1].set_title("Coef[17]")
axs[1, 1].plot(np.transpose(samp["sp_coef"][:, :, 17]))
# fig.legend()
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
    tf=obs_days,
)

# %%
# Generate incidence of new run
ihr = fitted_medians["ihr"]
ihr_mult = fitted_medians["ihr_mult"]

model_incidence = jnp.sum(output.ys[3], axis=4)
model_incidence_0 = jnp.diff(model_incidence[:, :, 0, 0], axis=0)
model_incidence_1 = jnp.sum(model_incidence, axis=(2, 3))
model_incidence_1 = jnp.diff(model_incidence_1, axis=0)
model_incidence_1 -= model_incidence_0

sim_incidence = model_incidence_0 * ihr + model_incidence_1 * ihr * ihr_mult

# %%
# Visualization
dates = [
    m.config.INIT_DATE + datetime.timedelta(days=x)
    for x in range(len(sim_incidence))
]
date_format = mdates.DateFormatter("%b\n%y")
beta_coef = [m.beta_coef(x) for x in range(obs_days)]
colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
fig, axs = plt.subplots(2, 1)
axs[0].xaxis.set_major_formatter(date_format)
axs[0].set_prop_cycle(cycler(color=colors))
axs[0].plot(
    dates, np.log10(sim_incidence), label=["0-17", "18-49", "50-64", "65+"]
)
axs[0].plot(
    dates,
    np.log10(obs_incidence),
    label=["0-17 (obs)", "18-49 (obs)", "50-64 (obs)", "65+ (obs)"],
    linestyle="dashed",
)
axs[0].set_title("Observed vs fitted")
axs[0].set_ylabel("Hospitalization (log10)")

axs[1].xaxis.set_major_formatter(date_format)
axs[1].plot(dates, beta_coef)
axs[1].set_ylabel("Beta multiplier")
fig.legend()
fig.set_size_inches(6, 9)
fig.set_dpi(300)
plt.show()

# %%
# Calculate and plot sim vs obs seroprevalence
never_infected = jnp.sum(output.ys[0][sero_days, :, 0, :, :], axis=(2, 3))
sim_seroprevalence = 1 - never_infected / m.config.POPULATION
# sim_seroprevalence = sim_seroprevalence
sim_lseroprevalence = jnp.log(sim_seroprevalence / (1 - sim_seroprevalence))
obs_seroprevalence = 1 / (1 + jnp.exp(-obs_sero_lmean))

# %%
fig, ax = plt.subplots(1)
ax.xaxis.set_major_formatter(date_format)
ax.set_prop_cycle(cycler(color=colors))
sero_dates = [
    pd.to_datetime("2020-07-26") + pd.Timedelta(d, unit="day")
    for d in sero_days.tolist()
]
ax.plot(
    sero_dates,
    sim_seroprevalence,
    label=["0-17", "18-49", "50-64", "65+"],
)
ax.plot(
    sero_dates,
    obs_seroprevalence,
    label=["0-17 (obs)", "18-49 (obs)", "50-64 (obs)", "65+ (obs)"],
    linestyle="dashed",
)
ax.set_title("Observed vs fitted")
ax.set_ylabel("Seroprevalence")
plt.legend()
plt.show()

# %%
print("Theoretical cumulative vax rate:")
vax = jnp.array([inferer.vaccination_rate(t) for t in range(250)])
print(jnp.sum(vax, axis=(0)))

print("Simulated vax compartment size (0, 1, 2):")
sim_vax = [jnp.sum(output.ys[i][-1,], axis=(1, 3)) for i in range(3)]
sim_vax = jnp.sum(jnp.array(sim_vax), axis=(0,))
print(sim_vax / inferer.config.POPULATION[:, None])
