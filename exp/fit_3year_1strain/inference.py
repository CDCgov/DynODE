import copy

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist


def infer_model(
    obs_incidence,
    obs_sero_lmean,
    obs_sero_lsd,
    sero_days,
    J,
    model,
    initializer,
):
    """
    Full model for inference (prior and likelihood), bypassing the use of
    inferer.infer()
    """
    m = copy.deepcopy(model)

    # Parameters
    ## coefficients for bspline, forcining first one to be 1.0
    with numpyro.plate("J", J):
        sp_coef = numpyro.sample(
            "sp_coef", dist.TruncatedNormal(1.0, 0.1, low=0.0, high=5.0)
        )
    m.config.BSPLINE_COEFFS = jnp.append(jnp.array([1.0]), sp_coef)

    ## R0
    # r0_dist = dist.TransformedDistribution(
    #     dist.Beta(50, 350), dist.transforms.AffineTransform(1.0, 4.0)
    # )
    # r0 = numpyro.sample("r0", r0_dist)
    r0 = numpyro.deterministic("r0", 1.0)
    m.config.STRAIN_R0s = jnp.array([r0])

    initial_inf_prop = numpyro.sample("INITIAL_INFECTIONS", dist.Beta(10, 990))
    initial_infections = initial_inf_prop * 0.1 * initializer.config.POP_SIZE

    # Obtain ODE solution
    sol = m.runner.run(
        initializer.load_initial_state(initial_infections),
        # m.INITIAL_STATE,
        args=m.get_parameters(),
        tf=len(obs_incidence),
    )

    # Simulated metrics
    ## Hospital incidence
    ## Calculate incidence and sample ihr and/or ihr multiplier
    ## ihr multiplier reduces severity for people with previous
    ## infection or vaccination
    model_incidence = jnp.sum(sol.ys[3], axis=4)
    model_incidence_0 = jnp.diff(model_incidence[:, :, 0, 0], axis=0)

    model_incidence_1 = jnp.sum(model_incidence, axis=(2, 3))
    model_incidence_1 = jnp.diff(model_incidence_1, axis=0)
    model_incidence_1 -= model_incidence_0

    with numpyro.plate("num_age", 4):
        ihr = numpyro.sample("ihr", dist.Beta(1, 9))

    ihr_mult = numpyro.sample("ihr_mult", dist.Beta(2000, 8000))
    # ihr_mult = numpyro.deterministic("ihr_mult", 1.0)

    sim_incidence = (
        model_incidence_0 * ihr + model_incidence_1 * ihr * ihr_mult
    )

    ## Seroprevalence
    never_infected = jnp.sum(sol.ys[0][sero_days, :, 0, :, :], axis=(2, 3))
    sim_seroprevalence = 1 - never_infected / m.config.POPULATION
    # sim_seroprevalence = sim_seroprevalence[:, :]
    sim_lseroprevalence = jnp.log(
        sim_seroprevalence / (1 - sim_seroprevalence)
    )

    # Observation model
    numpyro.sample(
        "incidence",
        dist.Poisson(sim_incidence),
        obs=obs_incidence,
    )

    numpyro.sample(
        "lseroprevalence",
        dist.Normal(sim_lseroprevalence, obs_sero_lsd),
        obs=obs_sero_lmean,
    )
