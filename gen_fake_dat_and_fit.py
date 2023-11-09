# %%
import jax.config
import jax.numpy as jnp
import numpy as np
import numpyro

from config.config_base import ConfigBase
from mechanistic_compartments import build_basic_mechanistic_model
from model_odes.seir_model_v5 import seirw_ode2  # , seirw_ode

# from jax.random import PRNGKey
# from numpyro.infer import MCMC, NUTS
# from code_fragments_deprecated.inference import infer_model

# Use 4 cores
numpyro.set_host_device_count(4)
jax.config.update("jax_enable_x64", True)

# True model
cb = ConfigBase()
cb.STRAIN_SPECIFIC_R0 = jnp.array([1.5, 2.5, 3.5])
model = build_basic_mechanistic_model(cb)
solution = model.run(seirw_ode2, tf=100, save=False)
model_incidence = jnp.sum(solution.ys[5], axis=2)
model_incidence = jnp.diff(model_incidence, axis=0)
rng = np.random.default_rng(seed=8675399)
m = np.asarray(model_incidence)
fake_obs = rng.poisson(m)

# %%
# Perform inference
# Reducing max_tree_depth here to reduce fitting time
# The new way of doing it shown below!

model.infer(
    seirw_ode2,
    fake_obs,
    sample_dist_dict={
        "infectious_period": numpyro.distributions.HalfCauchy(1.0)
    },
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
