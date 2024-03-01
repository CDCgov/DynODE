# %%
import os

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mechanistic_model.covid_initializer import CovidInitializer
from mechanistic_model.mechanistic_runner import MechanisticRunner
from mechanistic_model.static_value_parameters import StaticValueParameters
from model_odes.seip_model import seip_ode

# Paths to JSONs
EXP_ROOT_PATH = "exp/fit_2epochs_fake/"
GLOBAL_CONFIG_PATH = EXP_ROOT_PATH + "config_global.json"
INITIALIZER_CONFIG_PATH = EXP_ROOT_PATH + "config_initializer.json"
RUNNER_CONFIG_PATH = EXP_ROOT_PATH + "config_runner.json"

# %%
# Assembe components and run
initializer = CovidInitializer(INITIALIZER_CONFIG_PATH, GLOBAL_CONFIG_PATH)
runner = MechanisticRunner(seip_ode)

static_params = StaticValueParameters(
    initializer.get_initial_state(),
    RUNNER_CONFIG_PATH,
    GLOBAL_CONFIG_PATH,
)

output = runner.run(
    initializer.get_initial_state(), static_params.get_parameters(), tf=450
)

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
obs_df = pd.DataFrame(fake_obs)
obs_df.columns = ["age" + str(i) for i in range(4)]
obs_df["time"] = np.arange(450)
obs_df.to_csv(os.path.join(EXP_ROOT_PATH, "fake_dat.csv"), index=False)
