# %%
import copy

import jax.config
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import pandas as pd
from cycler import cycler
from inference import infer_model
from jax.random import PRNGKey
from numpyro.infer import MCMC, NUTS

from config.config_base import ConfigBase
from mechanistic_compartments import build_basic_mechanistic_model
from model_odes.seip_model import seip_ode

# Use 4 cores
numpyro.set_host_device_count(5)
jax.config.update("jax_enable_x64", True)
# %%
# Observations
obs_df = pd.read_csv("./data/hospitalization-data/hospital_220220_221105.csv")
obs_incidence = obs_df.groupby(["date"])["new_admission_7"].apply(np.array)
obs_incidence = jnp.array(obs_incidence.tolist())
obs_incidence = obs_incidence[0:200,]

fig, ax = plt.subplots(1)
ax.plot(np.asarray(obs_incidence), label=["0-17", "18-49", "50-64", "65+"])
fig.legend()
ax.set_title("Observed data")
plt.show()

# %%
# Config to US population sizes
pop = 3.28e8
init_inf_prop = 0.03
intro_perc = 0.02
cb = ConfigBase(
    POP_SIZE=pop,
    INFECTIOUS_PERIOD=7.0,
    INITIAL_INFECTIONS=init_inf_prop * pop,
    INTRODUCTION_PERCENTAGE=intro_perc,
    INTRODUCTION_SCALE=10,
    NUM_WANING_COMPARTMENTS=5,
)
model = build_basic_mechanistic_model(cb)
model.VAX_EFF_MATRIX = jnp.array(
    [
        [0, 0.29, 0.58],  # delta
        [0, 0.24, 0.48],  # omicron1
        [0, 0.19, 0.38],  # BA1.1
    ]
)


# %%
# MCMC specifications for "cold run"
nuts = NUTS(
    infer_model,
    dense_mass=True,
    max_tree_depth=7,
    init_strategy=numpyro.infer.init_to_median(),
    target_accept_prob=0.80,
    # find_heuristic_step_size=True,
)
mcmc = MCMC(
    nuts,
    num_warmup=500,
    num_samples=500,
    num_chains=5,
    progress_bar=True,
)

# %%
# mcmc.warmup(
#     rng_key=PRNGKey(8811967),
#     collect_warmup=True,
#     incidence=obs_incidence,
#     model=model,
# )

mcmc.run(
    rng_key=PRNGKey(8811968),
    incidence=obs_incidence,
    model=model,
)

# %%
samp = mcmc.get_samples(group_by_chain=True)
fig, axs = plt.subplots(2, 2)
axs[0, 0].set_title("Intro Time")
axs[0, 0].plot(np.transpose(samp["INTRO_TIME"]))
axs[0, 1].set_title("Intro Percentage")
axs[0, 1].plot(np.transpose(samp["INTRO_PERC"]))
axs[1, 0].set_title("R02")
axs[1, 0].plot(np.transpose(samp["r0_2"]))
axs[1, 1].set_title("R03")
axs[1, 1].plot(np.transpose(samp["r0_3"]), label=range(1, 6))
fig.legend()
plt.show()

# %%
# Take median of runs and check if fit is good
fitted_medians = {k: jnp.median(v[:, -1], axis=0) for k, v in samp.items()}
cb1 = ConfigBase(
    POP_SIZE=pop,
    INFECTIOUS_PERIOD=7.0,
    INITIAL_INFECTIONS=fitted_medians["INITIAL_INFECTIONS"],
    INTRODUCTION_PERCENTAGE=fitted_medians["INTRO_PERC"],
    INTRODUCTION_TIMES=[fitted_medians["INTRO_TIME"]],
    INTRODUCTION_SCALE=fitted_medians["INTRO_SCALE"],
    NUM_WANING_COMPARTMENTS=5,
)
cb1.STRAIN_SPECIFIC_R0 = jnp.append(
    jnp.array([1.2]),
    jnp.append(fitted_medians["r0_2"], fitted_medians["r0_3"]),
)
model1 = build_basic_mechanistic_model(cb1)
imm = fitted_medians["imm_factor"]
ihr1 = fitted_medians["ihr"]
ihr_mult = fitted_medians["ihr_mult"]
model1.CROSSIMMUNITY_MATRIX = jnp.array(
    [
        [
            0.0,  # 000
            1.0,  # 001
            1.0,  # 010
            1.0,  # 011
            1.0,  # 100
            1.0,  # 101
            1.0,  # 110
            1.0,  # 111
        ],
        [0.0, imm, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [
            0.0,
            imm**2,
            imm,
            imm,
            1.0,
            1.0,
            1.0,
            1.0,
        ],
    ]
)
model1.VAX_EFF_MATRIX = jnp.array(
    [
        [0, 0.29, 0.58],  # delta
        [0, 0.24, 0.48],  # omicron1
        [0, 0.19, 0.38],  # BA1.1
    ]
)

solution1 = model1.run(
    seip_ode,
    tf=250,
    show=True,
    save=False,
    # plot_commands=["S[:, 0, :, :]", "BA1.1", "omicron", "delta"],
    log_scale=True,
)

model_incidence = jnp.sum(solution1.ys[3], axis=4)
model_incidence_0 = jnp.diff(model_incidence[:, :, 0, 0], axis=0)

model_incidence_1 = jnp.sum(model_incidence, axis=(2, 3))
model_incidence_1 = jnp.diff(model_incidence_1, axis=0)
model_incidence_1 -= model_incidence_0

sim_incidence = model_incidence_0 * ihr1 + model_incidence_1 * ihr1 * ihr_mult

colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
fig, ax = plt.subplots(1)
ax.set_prop_cycle(cycler(color=colors))
ax.plot(sim_incidence, label=["0-17", "18-49", "50-64", "65+"])
ax.plot(
    obs_incidence,
    label=["0-17 (obs)", "18-49 (obs)", "50-64 (obs)", "65+ (obs)"],
    linestyle="dashed",
)
fig.legend()
ax.set_title("Observed vs fitted")
plt.show()

# %%
# Check covariance matrix
samp_all = copy.deepcopy(mcmc.get_samples(group_by_chain=False))
ihr_dict = {
    "ihr_" + str(i): v for i, v in enumerate(jnp.transpose(samp_all["ihr"]))
}
del samp_all["ihr"]
samp_all.update(ihr_dict)

samp_df = pd.DataFrame(samp_all)
samp_df.corr()
