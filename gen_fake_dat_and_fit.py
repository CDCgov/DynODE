# %%
import jax.config
import jax.numpy as jnp
import numpy as np
import numpyro
from inference import infer_model
from model_odes.seir_model_v5 import seirw_ode, seirw_ode2
from mechanistic_compartments import build_basic_mechanistic_model
from config.config_base import ConfigBase

# from inference import infer_model2
from numpyro.infer import NUTS, MCMC
from jax.random import PRNGKey

# Use 4 cores
numpyro.set_host_device_count(4)
jax.config.update("jax_enable_x64", True)

cb = ConfigBase()
cb.STRAIN_SPECIFIC_R0 = jnp.array([1.5, 2.5, 3.5])
model = build_basic_mechanistic_model(cb)
solution = model.run(
    seirw_ode2, tf=100, save=False
)  # save_path="output/example.png")
# solution2 = model.run(seirw_ode2, tf=100, save=False)

# %%
# args = model.get_args()
# state = model.INITIAL_STATE
# solution = seirw_ode(state, 1, args)
# solution2 = seirw_ode2(state, 1, args)

# %%
model_incidence = jnp.sum(solution.ys[5], axis=2)
model_incidence = jnp.diff(model_incidence, axis=0)
rng = np.random.default_rng(seed=8675399)
m = np.asarray(model_incidence)
k = 2
p = k / (k + m)
n = k

# %%
fake_obs = rng.poisson(m)
# fake_obs = rng.negative_binomial(n, p)
# print(fake_obs)

# %%
# Perform inference
args = model.get_args()
initial_state = model.INITIAL_STATE
mcmc = MCMC(
    NUTS(infer_model, dense_mass=True, max_tree_depth=5),
    num_warmup=200,
    num_samples=200,
    thinning=1,
    num_chains=4,
    progress_bar=True,
)
mcmc.run(
    PRNGKey(8675328),
    times=np.linspace(0.0, 100.0, 101),
    incidence=fake_obs,
    model=model,
)
mcmc.print_summary()
