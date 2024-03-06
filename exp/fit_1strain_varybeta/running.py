# %%
import types
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import os

os.chdir("/home/bentoh/projects/scenarios-2")
from mechanistic_model.covid_initializer import CovidInitializer
from mechanistic_model.mechanistic_runner import MechanisticRunner
from mechanistic_model.static_value_parameters import StaticValueParameters
from model_odes.seip_model import seip_ode

# Paths to JSONs
EXP_ROOT_PATH = "exp/fit_1strain_varybeta/"
ORI_GLOBAL_CONFIG_PATH = EXP_ROOT_PATH + "ori_config_global.json"
NEW_GLOBAL_CONFIG_PATH = EXP_ROOT_PATH + "new_config_global.json"
INITIALIZER_CONFIG_PATH = EXP_ROOT_PATH + "ori_config_initializer.json"
RUNNER_CONFIG_PATH = EXP_ROOT_PATH + "config_runner.json"

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
runner = MechanisticRunner(seip_ode)

static_params = StaticValueParameters(
    new_init_state,
    RUNNER_CONFIG_PATH,
    NEW_GLOBAL_CONFIG_PATH,
)


# %%
def deBoor(k, x, t, c, p):
    """Evaluates S(x).

    Arguments
    ---------
    k: Index of knot interval that contains x.
    x: Position.
    t: Array of knot positions, needs to be padded as described above.
    c: Array of control points.
    p: Degree of B-spline.
    """
    d = [c[j + k - p] for j in range(0, p + 1)]

    for r in range(1, p + 1):
        for j in range(p, r - 1, -1):
            alpha = (x - t[j + k - p]) / (t[j + 1 + k - r] - t[j + k - p])
            d[j] = (1.0 - alpha) * d[j - 1] + alpha * d[j]

    return d[p]


def custom_beta_coef(self, t):
    knots = self.config.BSPLINE_KNOTS
    coeffs = self.config.BSPLINE_COEFFS

    value = deBoor(
        jnp.searchsorted(knots, t, "right") - 1, t, knots, coeffs, 3
    )
    return value


static_params.beta_coef = types.MethodType(custom_beta_coef, static_params)

# %%
k = [
    0.0,
    0.0,
    0.0,
    0.0,
    30.0,
    60.0,
    90.0,
    120.0,
    150.0,
    180.0,
    210.0,
    240.0,
    270.0,
    300.0,
    300.0,
    300.0,
    300.0,
]

static_params.config.BSPLINE_KNOTS = jnp.array(k)
static_params.config.BSPLINE_COEFFS = jnp.array(
    [1.0] * 4 + [2.0] * 3 + [1.0] * 6
)

# %%
output = runner.run(new_init_state, static_params.get_parameters(), tf=290)

# %%
# Generate hospitalization data
ihr = [0.004, 0.014, 0.026, 0.228]
model_incidence = jnp.sum(output.ys[3], axis=(2, 3, 4))
model_incidence = jnp.diff(model_incidence, axis=0)
rng = np.random.default_rng(seed=8675309)
m = np.asarray(model_incidence) * ihr
fake_obs = rng.poisson(m)

# %%
# Visualization
fig, ax = plt.subplots(1)
ax.plot(fake_obs, label=["0-17", "18-49", "50-64", "65+"])
fig.legend()
ax.set_title("Observed data")
plt.show()

# %%
