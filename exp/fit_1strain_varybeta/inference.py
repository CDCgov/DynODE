import copy

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist


def infer_model(incidence, model):
    m = copy.deepcopy(model)

    # Parameters
    with numpyro.plate("J", 18):
        sp_coef = numpyro.sample(
            "sp_coef", dist.TruncatedNormal(1.0, 0.1, low=0.0, high=5.0)
        )
    m.config.BSPLINE_COEFFS = jnp.append(jnp.array([1.0]), sp_coef)
    # m.config.BSPLINE_COEFFS = sp_coef
    # r0_dist = dist.TransformedDistribution(
    #     dist.Beta(100, 300), dist.transforms.AffineTransform(1.0, 4.0)
    # )
    # r0 = numpyro.sample("r0", r0_dist)
    r0 = numpyro.deterministic("r0", 1.2)
    m.config.STRAIN_R0s = jnp.array([r0])

    sol = m.runner.run(
        m.INITIAL_STATE,
        args=m.get_parameters(),
        tf=len(incidence),
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
    ihr_mult = numpyro.deterministic("ihr_mult", 0.2)

    sim_incidence = (
        model_incidence_0 * ihr + model_incidence_1 * ihr * ihr_mult
    )

    numpyro.sample(
        "incidence",
        dist.Poisson(sim_incidence),
        obs=incidence,
    )


def infer_model_fake(incidence, model):
    m = copy.deepcopy(model)

    # Parameters
    with numpyro.plate("J", 18):
        sp_coef = numpyro.sample(
            "sp_coef", dist.TruncatedNormal(1.0, 0.1, low=0.0, high=5.0)
        )
    m.config.BSPLINE_COEFFS = jnp.append(jnp.array([1.0]), sp_coef)
    # m.config.BSPLINE_COEFFS = sp_coef
    r0_dist = dist.TransformedDistribution(
        dist.Beta(100, 300), dist.transforms.AffineTransform(1.0, 4.0)
    )
    r0 = numpyro.sample("r0", r0_dist)
    # r0 = numpyro.deterministic("r0", 1.2)
    m.config.STRAIN_R0s = jnp.array([r0])

    sol = m.runner.run(
        m.INITIAL_STATE,
        args=m.get_parameters(),
        tf=len(incidence),
    )

    model_incidence = jnp.sum(sol.ys[3], axis=4)
    model_incidence_0 = jnp.diff(model_incidence[:, :, 0, 0], axis=0)

    model_incidence_1 = jnp.sum(model_incidence, axis=(2, 3))
    model_incidence_1 = jnp.diff(model_incidence_1, axis=0)
    model_incidence_1 -= model_incidence_0

    with numpyro.plate("num_age", 4):
        ihr = numpyro.sample("ihr", dist.Beta(1, 9))

    # IHR multiplier is very correlated with IHR (duh, might be better fixed)
    ihr_mult = numpyro.sample("ihr_mult", dist.Beta(100, 900))
    # ihr_mult = numpyro.deterministic("ihr_mult", 0.2)

    sim_incidence = (
        model_incidence_0 * ihr + model_incidence_1 * ihr * ihr_mult
    )

    numpyro.sample(
        "incidence",
        dist.Poisson(sim_incidence),
        obs=incidence,
    )