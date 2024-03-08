# %%
import copy
import types

import jax.config
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
from jax.random import PRNGKey
from numpyro.infer import MCMC, NUTS

from exp.fit_1strain_varybeta.inference import infer_model_fake
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
inferer.config.BSPLINE_COEFFS = jnp.array([1.0] * 10 + [2.0] * 5 + [1.0] * 4)
inferer.config.STRAIN_R0s = [2.0]

# %%
# Create fake data based on "true" values and visualize
true_r0 = 2.5
true_ihr = jnp.array([0.01, 0.02, 0.03, 0.2])
true_ihr_mult = 0.25
true_bspline_coefs = [
    1.0,
    1.5,
    1.8,
    2.0,
    2.2,
    2.0,
    1.8,
    1.5,
    1.3,
    1.3,
    1.6,
    1.9,
    1.8,
    1.8,
    1.6,
    1.5,
    1.4,
    1.3,
    1.2,
]

m = copy.deepcopy(inferer)
m.config.BSPLINE_COEFFS = jnp.array(true_bspline_coefs)
m.config.STRAIN_R0s = [true_r0]
output = runner.run(new_init_state, m.get_parameters(), tf=450)

model_incidence = jnp.sum(output.ys[3], axis=4)
model_incidence_0 = jnp.diff(model_incidence[:, :, 0, 0], axis=0)
model_incidence_1 = jnp.sum(model_incidence, axis=(2, 3))
model_incidence_1 = jnp.diff(model_incidence_1, axis=0)
model_incidence_1 -= model_incidence_0

sim_incidence = (
    model_incidence_0 * true_ihr + model_incidence_1 * true_ihr * true_ihr_mult
)

rng = np.random.default_rng(seed=8675309)
fake_obs = rng.poisson(sim_incidence)

fig, ax = plt.subplots(1)
ax.plot(fake_obs, label=["0-17", "18-49", "50-64", "65+"])
fig.legend()
ax.set_title("Observed data")
plt.show()

# %%
# MCMC
nuts = NUTS(
    infer_model_fake,
    dense_mass=True,
    max_tree_depth=6,
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
mcmc.run(
    rng_key=PRNGKey(8811968),
    incidence=fake_obs,
    model=inferer,
)
mcmc.print_summary()

# %%
# Extract samples
samp = mcmc.get_samples(group_by_chain=True)
fitted_medians_chain = {k: jnp.median(v, axis=1) for k, v in samp.items()}
fitted_medians = {k: jnp.median(v[:, -1], axis=0) for k, v in samp.items()}

# %%
# Compare fitted numbers vs "true" numbers
fitted_coef = np.asarray(fitted_medians["sp_coef"])
fitted_coef = np.append([1.0], fitted_coef)

fig, ax = plt.subplots(1)
ax.scatter(
    np.asarray(m.config.BSPLINE_COEFFS) * true_r0,
    fitted_coef * fitted_medians["r0"],
)
plt.axline((2.5, 2.5), (3.0, 3.0))
plt.show()
