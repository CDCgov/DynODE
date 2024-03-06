# %%
import copy
import datetime
import os
import types
from functools import partial

import jax.config
import jax.numpy as jnp
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import pandas as pd

os.chdir("/home/bentoh/projects/scenarios-2")
import jax.numpy as jnp
import numpy as np
from cycler import cycler
from jax.random import PRNGKey
from numpyro import distributions as Dist
from numpyro.infer import MCMC, NUTS

from config.config import Config
from exp.fit_1strain_varybeta.inference import infer_model_fake
from mechanistic_model.abstract_initializer import MechanisticInitializer

# %%
from mechanistic_model.covid_initializer import CovidInitializer
from mechanistic_model.mechanistic_inferer import MechanisticInferer
from mechanistic_model.mechanistic_runner import MechanisticRunner
from mechanistic_model.solution_iterpreter import SolutionInterpreter
from mechanistic_model.static_value_parameters import StaticValueParameters

# from code_fragments_deprecated.config_base import ConfigBase
# from mechanistic_compartments import build_basic_mechanistic_model
from model_odes.seip_model import seip_ode

EXP_ROOT_PATH = "exp/fit_1strain_varybeta/"
ORI_GLOBAL_CONFIG_PATH = EXP_ROOT_PATH + "ori_config_global.json"
NEW_GLOBAL_CONFIG_PATH = EXP_ROOT_PATH + "new_config_global.json"
INITIALIZER_CONFIG_PATH = EXP_ROOT_PATH + "ori_config_initializer.json"
RUNNER_CONFIG_PATH = EXP_ROOT_PATH + "config_runner.json"
INFERER_CONFIG_PATH = EXP_ROOT_PATH + "config_inferer.json"
# INTERPRETER_CONFIG_PATH = EXP_ROOT_PATH + "config_interpreter.json"

numpyro.set_host_device_count(5)
jax.config.update("jax_enable_x64", True)

# %%
# Take original initializer and create new 1 strain state
initializer = CovidInitializer(INITIALIZER_CONFIG_PATH, ORI_GLOBAL_CONFIG_PATH)
ori_init_state = initializer.get_initial_state()

new_init_state = []
for st in ori_init_state:
    shp = list(st.shape)
    shp[1] = 2
    shp[-1] = 1 if shp[-1] == 3 else shp[-1]
    shp = tuple(shp)
    newst = jnp.zeros(shp)
    if shp[-1] == 1:
        st_1strain = jnp.sum(st, axis=-1)
        newst = newst.at[:, 0, :, 0].set(st_1strain[:, 0, :])
        newst = newst.at[:, 1, :, 0].set(jnp.sum(st_1strain[:, 1:, :], axis=1))
    else:
        newst = newst.at[:, 0, :, :].set(st[:, 0, :, :])
        newst = newst.at[:, 1, :, :].set(jnp.sum(st[:, 1:, :, :], axis=1))
    new_init_state = new_init_state + [newst]

new_init_state = tuple(new_init_state)

# %%
# Config to US population sizes
runner = MechanisticRunner(seip_ode)
inferer = MechanisticInferer(
    NEW_GLOBAL_CONFIG_PATH,
    INFERER_CONFIG_PATH,
    runner,
    new_init_state,
)


# %%
@jax.jit
def deBoor(k, x, t, c):
    """Evaluates S(x).

    Arguments
    ---------
    k: Index of knot interval that contains x.
    x: Position.
    t: Array of knot positions, needs to be padded as described above.
    c: Array of control points.
    p: Degree of B-spline.
    """
    d = [c[j + k - 3] for j in range(0, 3 + 1)]

    for r in range(1, 3 + 1):
        for j in range(3, r - 1, -1):
            alpha = (x - t[j + k - 3]) / (t[j + 1 + k - r] - t[j + k - 3])
            d[j] = (1.0 - alpha) * d[j - 1] + alpha * d[j]

    return d[3]


@partial(jax.jit, static_argnums=(0))
def custom_beta_coef(self, t):
    knots = self.config.BSPLINE_KNOTS
    coeffs = self.config.BSPLINE_COEFFS

    value = deBoor(jnp.searchsorted(knots, t, "right") - 1, t, knots, coeffs)
    return value


inferer.beta_coef = types.MethodType(custom_beta_coef, inferer)

# %%
k = list(np.arange(0.0, 480.0 + 1, 30.0))
k = [0.0] * 3 + k + [480.0] * 3

inferer.config.BSPLINE_KNOTS = jnp.array(k)
inferer.config.BSPLINE_COEFFS = jnp.array([1.0] * 10 + [2.0] * 5 + [1.0] * 4)
inferer.config.STRAIN_R0s = [2.0]

# %%
true_r0 = 2.5
true_ihr = [0.01, 0.02, 0.03, 0.2]
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

# %%
rng = np.random.default_rng(seed=8675309)
fake_obs = rng.poisson(sim_incidence)

fig, ax = plt.subplots(1)
ax.plot(fake_obs, label=["0-17", "18-49", "50-64", "65+"])
fig.legend()
ax.set_title("Observed data")
plt.show()

# %%
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
    num_chains=3,
    progress_bar=True,
)
mcmc.run(
    rng_key=PRNGKey(8811968),
    incidence=fake_obs,
    model=inferer,
)
mcmc.print_summary()

# %%
samp = mcmc.get_samples(group_by_chain=True)
fitted_medians_chain = {k: jnp.median(v, axis=1) for k, v in samp.items()}
fitted_medians = {k: jnp.median(v[:, -1], axis=0) for k, v in samp.items()}

# %%
fitted_coef = np.asarray(fitted_medians["sp_coef"])
fitted_coef = np.append([1.0], fitted_coef)
fitted_coef

# %%
fig, ax = plt.subplots(1)
ax.scatter(
    np.asarray(m.config.BSPLINE_COEFFS) * true_r0,
    fitted_coef * fitted_medians["r0"],
)
plt.axline((2.0, 2.0), (2.3, 2.3))
plt.show()
