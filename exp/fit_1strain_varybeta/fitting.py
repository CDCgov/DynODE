# %%
import copy
import datetime
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

from exp.fit_1strain_varybeta.inference import infer_model
from exp.fit_1strain_varybeta.utilities import (
    custom_beta_coef,
    make_1strain_init_state,
)
from mechanistic_model.covid_initializer import CovidInitializer
from mechanistic_model.mechanistic_inferer import MechanisticInferer
from mechanistic_model.mechanistic_runner import MechanisticRunner
from model_odes.seip_model import seip_ode

numpyro.set_host_device_count(5)
jax.config.update("jax_enable_x64", True)

# Paths
EXP_ROOT_PATH = "exp/fit_1strain_varybeta/"
ORI_GLOBAL_CONFIG_PATH = EXP_ROOT_PATH + "ori_config_global.json"
NEW_GLOBAL_CONFIG_PATH = EXP_ROOT_PATH + "new_config_global.json"
INITIALIZER_CONFIG_PATH = EXP_ROOT_PATH + "ori_config_initializer.json"
RUNNER_CONFIG_PATH = EXP_ROOT_PATH + "config_runner.json"
INFERER_CONFIG_PATH = EXP_ROOT_PATH + "config_inferer.json"

# %%
# Observations
## Incidence
obs_df = pd.read_csv("./data/hospitalization-data/hospital_220220_231231.csv")
obs_incidence = obs_df.groupby(["date"])["new_admission_7"].apply(np.array)
obs_incidence = jnp.array(obs_incidence.tolist())
obs_incidence = obs_incidence[:450]

fig, ax = plt.subplots(1)
ax.plot(np.asarray(obs_incidence), label=["0-17", "18-49", "50-64", "65+"])
fig.legend()
ax.set_title("Observed data")
plt.show()

# %%
# Seroprevalence
sero_df = pd.read_csv("./data/serological-data/donor2022.csv")
sero_days = [5, 95, 185, 275]
obs_sero_lmean = sero_df.groupby(["date"])["logit_mean"].apply(np.array)
obs_sero_lmean = jnp.array(obs_sero_lmean.to_list())
obs_sero_lsd = sero_df.groupby(["date"])["logit_sd"].apply(np.array)
obs_sero_lsd = jnp.array(obs_sero_lsd.to_list())

# %%
# Take original initializer and create new 1 strain state
initializer = CovidInitializer(INITIALIZER_CONFIG_PATH, ORI_GLOBAL_CONFIG_PATH)
ori_init_state = initializer.get_initial_state()
new_init_state = make_1strain_init_state(ori_init_state)

# %%
# Create runner and inferer, override beta_coef function
runner = MechanisticRunner(seip_ode)
inferer = MechanisticInferer(
    NEW_GLOBAL_CONFIG_PATH,
    INFERER_CONFIG_PATH,
    runner,
    new_init_state,
)

inferer.beta_coef = types.MethodType(custom_beta_coef, inferer)

# %%
# Set default bspline knots and coefficients
k = list(np.arange(0.0, 480.0 + 1, 30.0))
k = [0.0] * 3 + k + [480.0] * 3

inferer.config.BSPLINE_KNOTS = jnp.array(k)
inferer.config.BSPLINE_COEFFS = jnp.array([1.0] * 10 + [2.0] * 3 + [1.0] * 6)

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
    obs_sero_lsd=obs_sero_lsd / 10,  # / 10 to increase weigtage
    sero_days=sero_days,
    model=inferer,
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
output = runner.run(new_init_state, m.get_parameters(), tf=450)

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
date_format = mdates.DateFormatter("%b %y")
beta_coef = [m.beta_coef(x) for x in range(450)]
colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
fig, axs = plt.subplots(2, 1)
axs[0].xaxis.set_major_formatter(date_format)
axs[0].set_prop_cycle(cycler(color=colors))
axs[0].plot(dates, sim_incidence, label=["0-17", "18-49", "50-64", "65+"])
axs[0].plot(
    dates,
    obs_incidence,
    label=["0-17 (obs)", "18-49 (obs)", "50-64 (obs)", "65+ (obs)"],
    linestyle="dashed",
)
axs[0].set_title("Observed vs fitted")
axs[0].set_ylabel("Hospitalization")

axs[1].xaxis.set_major_formatter(date_format)
axs[1].plot(dates, beta_coef)
axs[1].set_ylabel("Beta multiplier")
fig.legend()
fig.set_size_inches(6, 6)
fig.set_dpi(300)
plt.show()

# %%
# Calculate and plot sim vs obs seroprevalence
never_infected = jnp.sum(output.ys[0][sero_days, :, 0, :, :], axis=(2, 3))
sim_seroprevalence = 1 - never_infected / m.config.POPULATION
sim_seroprevalence = sim_seroprevalence[:, 1:]
sim_lseroprevalence = jnp.log(sim_seroprevalence / (1 - sim_seroprevalence))
obs_seroprevalence = 1 / (1 + jnp.exp(-obs_sero_lmean))

# %%
quarters = np.arange(1, 5)
fig, ax = plt.subplots(1)
# ax[0].xaxis.set_major_formatter(date_format)
ax.set_prop_cycle(cycler(color=colors[1:]))
ax.plot(quarters, sim_seroprevalence, label=["18-49", "50-64", "65+"])
ax.plot(
    quarters,
    obs_seroprevalence,
    label=["18-49 (obs)", "50-64 (obs)", "65+ (obs)"],
    linestyle="dashed",
)
ax.set_title("Observed vs fitted")
ax.set_ylabel("Seroprevalence")
ax.set_xlabel("Quarter (2022)")
fig.legend()
plt.show()
