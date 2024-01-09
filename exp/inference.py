import copy

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from mechanistic_compartments import BasicMechanisticModel
from model_odes.seip_model import seip_ode


def infer_model(incidence, model: BasicMechanisticModel):
    m = copy.deepcopy(model)

    # Parameters
    with numpyro.plate("num_strains", 3):
        excess_r0 = numpyro.sample("excess_r0", dist.Beta(1, 9)) * 9
        r0 = numpyro.deterministic("r0", 1 + excess_r0)

    inf_period_scale = numpyro.sample(
        "inf_period_scale",
        dist.Beta(4, 6),
    )
    infectious_period = numpyro.deterministic(
        "INFECTIOUS_PERIOD", 3 + inf_period_scale * 17
    )

    intro_time_scale = numpyro.sample("intro_time_scale", dist.Beta(3, 9))
    introduction_time = numpyro.deterministic(
        "INTRO_TIME", intro_time_scale * 100
    )
    # introduction_perc = numpyro.sample("INTRO_PERCENTAGE", dist.Beta(0.1, 10))

    m.STRAIN_SPECIFIC_R0 = r0
    m.INFECTIOUS_PERIOD = infectious_period
    m.INTRODUCTION_TIMES_SAMPLE = [introduction_time]
    # m.INTRODUCTION_PERCENTAGE = introduction_perc

    sol = m.run(seip_ode, tf=len(incidence))
    model_incidence = jnp.sum(sol.ys[3], axis=(2, 3, 4))
    model_incidence = jnp.diff(model_incidence, axis=0)

    with numpyro.plate("num_age", 4):
        ihr = numpyro.sample("ihr", dist.Beta(0.1, 10))

    numpyro.sample(
        "incidence",
        dist.Poisson(model_incidence * ihr),
        obs=incidence,
    )
