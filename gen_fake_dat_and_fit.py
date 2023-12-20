# %%
import jax.config
import jax.numpy as jnp
import numpy as np
import numpyro
import matplotlib.pyplot as plt

from config.config_base import ConfigBase
from mechanistic_compartments import build_basic_mechanistic_model
from model_odes.seip_model import seip_ode  # , seirw_ode

# from jax.random import PRNGKey
# from numpyro.infer import MCMC, NUTS
# from code_fragments_deprecated.inference import infer_model

# Use 4 cores
numpyro.set_host_device_count(4)
jax.config.update("jax_enable_x64", True)

# %%
# True model
cb = ConfigBase(
    POP_SIZE=1e8,
    INITIAL_INFECTIONS=1e6,
    INTRODUCTION_PERCENTAGE=0.01,
    INTRODUCTION_TIMES=[60],
)
cb.STRAIN_SPECIFIC_R0 = jnp.array([1.5, 2.5, 2.5], dtype=jnp.float32)  # R0s
model = build_basic_mechanistic_model(cb)
ihr = [0.002, 0.004, 0.008, 0.06]

solution = model.run(seip_ode, tf=300, show=True, save=False)
model_incidence = jnp.sum(solution.ys[3], axis=(2, 3, 4))
model_incidence = jnp.diff(model_incidence, axis=0)
rng = np.random.default_rng(seed=8675399)
m = np.asarray(model_incidence) * ihr
k = 10.0
p = k / (k + m)
fake_obs = rng.negative_binomial(k, p)

# %%
fig, ax = plt.subplots(1)
ax.plot(m, label=[1, 2, 3, 4])
fig.legend()
plt.show()

# %%
# Perform inference
# Reducing max_tree_depth here to reduce fitting time
# The new way of doing it shown below!
model.MCMC_NUM_WARMUP = 100
model.MCMC_NUM_SAMPLES = 100
model.MCMC_NUM_CHAINS = 1
model.infer(
    seip_ode,
    fake_obs,
    negbin=True,
)

# mcmc = MCMC(
#     NUTS(infer_model, dense_mass=True, max_tree_depth=5),
#     num_warmup=500,
#     num_samples=500,
#     thinning=1,
#     num_chains=4,
#     progress_bar=True,
# )
# mcmc.run(
#     PRNGKey(8675328),
#     times=np.linspace(0.0, 100.0, 101),
#     incidence=fake_obs,
#     model=model,
# )
# mcmc.print_summary()
