import copy

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from mechanistic_compartments import BasicMechanisticModel
from model_odes.seip_model import seip_ode


def infer_model(incidence, model: BasicMechanisticModel):
    m = copy.deepcopy(model)

    # Parameters
    r0_2_dist = dist.TransformedDistribution(
        dist.Beta(50, 750), dist.transforms.AffineTransform(2.0, 8.0)
    )
    r0_3_dist = dist.TransformedDistribution(
        dist.Beta(2, 14), dist.transforms.AffineTransform(2.0, 8.0)
    )

    r0_2 = numpyro.sample("r0_2", r0_2_dist)
    # r0_2 = numpyro.deterministic("r0_2", 2.5)
    r0_3 = numpyro.sample("r0_3", r0_3_dist)

    introduction_time_dist = dist.TransformedDistribution(
        dist.Beta(30, 70), dist.transforms.AffineTransform(0.0, 100)
    )
    introduction_scale_dist = dist.TransformedDistribution(
        dist.Beta(50, 50), dist.transforms.AffineTransform(5.0, 10.0)
    )
    introduction_time = numpyro.sample("INTRO_TIME", introduction_time_dist)
    introduction_perc = numpyro.sample("INTRO_PERC", dist.Beta(20, 980))
    introduction_scale = numpyro.sample("INTRO_SCALE", introduction_scale_dist)

    # Very correlated with R0_3 (might be better fixed than estimated)
    imm = numpyro.sample("imm_factor", dist.Beta(700, 300))
    # imm = numpyro.deterministic("imm_factor", 0.7)

    m.STRAIN_SPECIFIC_R0 = jnp.array([1.2, r0_2, r0_3])
    m.INTRODUCTION_TIMES_SAMPLE = [introduction_time]
    m.INTRODUCTION_PERCENTAGE = introduction_perc
    m.INTRODUCTION_SCALE = introduction_scale
    m.CROSSIMMUNITY_MATRIX = jnp.array(
        [
            [
                0.0,  # 000
                1.0,  # 001
                1.0,  # 010
                1.0,  # 011
                1.0,  # 100
                1.0,  # 101
                1.0,  # 110
                1.0,  # 111
            ],
            [0.0, imm, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [
                0.0,
                imm**2,
                imm,
                imm,
                1.0,
                1.0,
                1.0,
                1.0,
            ],
        ]
    )

    sol = m.run(
        seip_ode,
        tf=len(incidence),
        sample_dist_dict={
            "INITIAL_INFECTIONS": dist.TransformedDistribution(
                dist.Beta(30, 970),
                dist.transforms.AffineTransform(0.0, model.POP_SIZE),
            )
        },
    )
    model_incidence = jnp.sum(sol.ys[3], axis=4)
    model_incidence_0 = jnp.diff(model_incidence[:, :, 0, 0], axis=0)

    model_incidence_1 = jnp.sum(model_incidence, axis=(2, 3))
    model_incidence_1 = jnp.diff(model_incidence_1, axis=0)
    model_incidence_1 -= model_incidence_0

    with numpyro.plate("num_age", 4):
        ihr = numpyro.sample("ihr", dist.Beta(1, 9))

    # IHR multiplier is very correlated with IHR (duh, might be better fixed)
    # ihr_mult = numpyro.sample("ihr_mult", dist.Beta(100, 900))
    ihr_mult = numpyro.deterministic("ihr_mult", 0.15)

    sim_incidence = (
        model_incidence_0 * ihr + model_incidence_1 * ihr * ihr_mult
    )

    numpyro.sample(
        "incidence",
        dist.Poisson(sim_incidence),
        obs=incidence,
    )
