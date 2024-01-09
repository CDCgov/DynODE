# %%
import pandas as pd
import jax.config
import jax.numpy as jnp
import numpy as np
import numpyro
import matplotlib.pyplot as plt

from config.config_base import ConfigBase
from mechanistic_compartments import build_basic_mechanistic_model
from model_odes.seip_model import seip_ode
from inference import infer_model

from jax.random import PRNGKey
from numpyro.infer import MCMC, NUTS

# Use 4 cores
numpyro.set_host_device_count(5)
jax.config.update("jax_enable_x64", True)
# %%
# Observations
obs_df = pd.read_csv("./data/hospitalization-data/hospital_220220_221105.csv")
obs_incidence = obs_df.groupby(["date"])["new_admission_7"].apply(np.array)
obs_incidence = jnp.array(obs_incidence.tolist())

fig, ax = plt.subplots(1)
ax.plot(np.asarray(obs_incidence), label=["0-17", "18-49", "50-64", "65+"])
fig.legend()
ax.set_title("Observed data")
plt.show()

# %%
# Config to US population sizes
pop = 328239523
cb = ConfigBase(
    POP_SIZE=pop,
    INITIAL_INFECTIONS=0.01 * pop,
    INTRODUCTION_PERCENTAGE=0.005,
    INTRODUCTION_TIMES=[60],
)
model = build_basic_mechanistic_model(cb)
# ihr = [0.002, 0.004, 0.008, 0.06]

# %%
# MCMC specifications for "cold run"
nuts = NUTS(
    infer_model,
    dense_mass=True,
    max_tree_depth=6,
    init_strategy=numpyro.infer.init_to_median(),
    target_accept_prob=0.70,
    find_heuristic_step_size=True,
    # regularize_mass_matrix=False,
)
mcmc = MCMC(
    nuts,
    num_warmup=500,
    num_samples=500,
    num_chains=5,
    progress_bar=True,
)

# %%
mcmc.warmup(
    rng_key=PRNGKey(8811977),
    collect_warmup=True,
    incidence=obs_incidence,
    model=model,
)

# %%
# Using last state of MCMC and samples to initialize the actual run
adapt_state = getattr(mcmc.last_state, "adapt_state")
step_sizes = getattr(adapt_state, "step_size")
inv_mass_matrices = getattr(adapt_state, "inverse_mass_matrix")
samp = mcmc.get_samples(group_by_chain=True)

# %%
init_dict = {k: jnp.median(v, axis=(0, 1)) for k, v in samp.items()}
init_step = jnp.median(step_sizes)
ind = np.argmin(np.where(step_sizes == init_step))
init_inv_mass = {k: v[ind,] for k, v in inv_mass_matrices.items()}

# %%
# "Actual run"
nuts = NUTS(
    infer_model,
    step_size=init_step,
    inverse_mass_matrix=init_inv_mass,
    dense_mass=True,
    max_tree_depth=6,
    init_strategy=numpyro.infer.init_to_value(values=init_dict),
    target_accept_prob=0.70,
    # find_heuristic_step_size=True,
    # regularize_mass_matrix=False,
)
mcmc = MCMC(
    nuts,
    num_warmup=1000,
    num_samples=1000,
    num_chains=5,
    progress_bar=True,
)

# %%
# Warm up (separating this so it's easy to diagnose)
mcmc.warmup(
    rng_key=PRNGKey(8811975),
    collect_warmup=True,
    incidence=obs_incidence,
    model=model,
)

# %%
# Actual sampling from actual run
mcmc.run(
    rng_key=PRNGKey(8811964),
    incidence=obs_incidence,
    model=model,
)

# %%
# Output and trace plots
mcmc.print_summary(exclude_deterministic=False)
samp = mcmc.get_samples(group_by_chain=True)

# %%
fig, axs = plt.subplots(2, 2)
axs[0, 0].set_title("Intro Time")
axs[0, 0].plot(np.transpose(samp["INTRO_TIME"]))
axs[0, 1].set_title("Infectious Period")
axs[0, 1].plot(np.transpose(samp["INFECTIOUS_PERIOD"]))
axs[1, 0].set_title("R02")
axs[1, 0].plot(np.transpose(samp["r0"][:, :, 1]))
axs[1, 1].set_title("R03")
axs[1, 1].plot(np.transpose(samp["r0"][:, :, 2]), label=range(1, 6))
fig.legend()
plt.show()

# %%
# Compare fitted medians with observed data
fitted_medians = {k: jnp.median(v, axis=(0, 1)) for k, v in samp.items()}
cb1 = ConfigBase(
    POP_SIZE=pop,
    INITIAL_INFECTIONS=0.01 * pop,
    INTRODUCTION_PERCENTAGE=0.005,
    INTRODUCTION_TIMES=[fitted_medians["INTRO_TIME"]],
)
cb1.STRAIN_SPECIFIC_R0 = fitted_medians["r0"]
cb1.INFECTIOUS_PERIOD = fitted_medians["INFECTIOUS_PERIOD"]
model1 = build_basic_mechanistic_model(cb1)
ihr1 = fitted_medians["ihr"]

solution1 = model1.run(
    seip_ode, tf=len(obs_incidence) - 1, show=True, save=False
)
model_incidence1 = jnp.sum(solution1.ys[3], axis=(2, 3, 4))
model_incidence1 = jnp.diff(model_incidence1, axis=0)
sim_incidence = np.asarray(model_incidence1) * ihr1

fig, ax = plt.subplots(1)
ax.plot(sim_incidence, label=["0-17", "18-49", "50-64", "65+"])
ax.plot(
    obs_incidence,
    label=["0-17 (obs)", "18-49 (obs)", "50-64 (obs)", "65+ (obs)"],
    linestyle="dashed",
)
fig.legend()
ax.set_title("Observed vs fitted")
plt.show()
