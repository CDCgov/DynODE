import copy

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from mechanistic_compartments import BasicMechanisticModel
from model_odes.seir_model_v5 import seirw_ode2


def infer_model(times, incidence, model: BasicMechanisticModel):
    # m = model
    m = copy.deepcopy(model)
    # Parameters
    # with numpyro.plate("num_strains", model.NUM_STRAINS):
    #     excess_r0 = numpyro.sample("excess_r0", dist.Exponential(1.0))
    #     r0 = numpyro.deterministic("r0", 1 + excess_r0)
    excess_r0 = numpyro.sample("excess_r0", dist.Exponential(1.0))
    r0 = numpyro.deterministic("r0", 1 + excess_r0)

    infectious_period = numpyro.sample(
        "infectious_period", dist.HalfCauchy(1.0)
    )
    waning_time = numpyro.sample("waning_time", dist.HalfCauchy(1.0))
    e_to_i = numpyro.sample("e_to_i", dist.HalfCauchy(1.0))

    m.STRAIN_SPECIFIC_R0 = jnp.asarray([1.5, 2.5, r0])
    m.INFECTIOUS_PERIOD = infectious_period
    m.EXPOSED_TO_INFECTIOUS = e_to_i
    m.WANING_TIME = waning_time

    sol = m.run(
        seirw_ode2, tf=100, plot=False, save=False
    )  # save_path="output/example.png")
    model_incidence = jnp.sum(sol.ys[5], axis=2)
    model_incidence = jnp.diff(model_incidence, axis=0)

    numpyro.sample("incidence", dist.Poisson(model_incidence), obs=incidence)
