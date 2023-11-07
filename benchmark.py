# %%
import jax.config
import jax.numpy as jnp
import numpy as np
import numpyro
from inference import infer_model
from model_odes.seir_model_v5 import seirw_ode, seirw_ode2
from mechanistic_compartments import build_basic_mechanistic_model
from config.config_base import ConfigBase
import timeit
from numpyro.infer import NUTS, MCMC
from jax.random import PRNGKey

# Use 4 cores
numpyro.set_host_device_count(4)
jax.config.update("jax_enable_x64", True)

cb = ConfigBase()
cb.STRAIN_SPECIFIC_R0 = jnp.array([1.5, 2.5, 3.5])
model = build_basic_mechanistic_model(cb)


def func1():
    return model.run(seirw_ode, tf=100, plot=False, save=False)


def func2():
    return model.run(seirw_ode2, tf=100, plot=False, save=False)


num_runs = 20
duration1 = timeit.Timer(func2).timeit(number=num_runs)
avg_duration1 = duration1 / num_runs
duration2 = timeit.Timer(func1).timeit(number=num_runs)
avg_duration2 = duration2 / num_runs

print(f"func1: On average it took {avg_duration1} seconds")
print(f"func2: On average it took {avg_duration2} seconds")
