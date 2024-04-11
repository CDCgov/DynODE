import copy

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from exp.fit_2year_1strain.utilities import deBoor


def infer_model(
    obs_incidence,
    obs_days,
    obs_sero_lmean,
    obs_sero_lsd,
    sero_days,
    J,
    model_day,
    ihr_knots,
    K,
    model,
    initializer,
):
    """
    Full model for inference (prior and likelihood), bypassing the use of
    inferer.infer()

    Parameters
    ----------
    obs_incidence: ndarray of observed incidence of shape (number of observation days,
    number of age groups)
    obs_days: ndarray of corresponding days of obs_incidence, e.g., [6, 13, ...]
    corresponding to observation of 6th, 13th and ... day of simulation.
    obs_sero_lmean: ndarray of observed mean seroprevalence (logit transformed)
    obs_sero_lsd: ndarray of standard deviations of observed seroprevalence. Note:
    this is sd for the logit transformed seroprevalence.
    sero_days: ndarray of corresponding days of observed seroprevalence.
    J: Number of coefficients for the beta multiplier bspline to fit
    model_day: total number of days to model
    ihr_knots: the knots for the IHR bspline
    K: Number of coefficients for the IHR multiplier bspline to fit
    model: a MechanisticInferer object
    initializer: a CovidInitializer object

    """
    m = copy.deepcopy(model)

    # Parameters
    ## coefficients for beta bspline, forcing first one to be 1.0
    with numpyro.plate("J", J):
        beta_coef = numpyro.sample(
            "beta_coef", dist.TruncatedNormal(1.0, 0.5, low=0.5, high=4.0)
        )
    m.config.BSPLINE_COEFFS = jnp.append(jnp.array([1.0]), beta_coef)

    ## R0
    r0 = numpyro.deterministic("r0", 1.0)
    m.config.STRAIN_R0s = jnp.array([r0])

    ## Initial infections & initial age proportion
    ini = copy.deepcopy(initializer)
    initial_inf_prop = numpyro.deterministic("INITIAL_INFECTIONS", 0.045)
    # initial_age_prop = numpyro.sample(
    #     "INITIAL_AGE_PROP", dist.Dirichlet(jnp.ones(4))
    # )
    initial_infections = initial_inf_prop * 0.1 * ini.config.POP_SIZE

    # Obtain ODE solution
    sol = m.runner.run(
        ini.load_initial_state(initial_infections),
        args=m.get_parameters(),
        tf=model_day,
    )

    # Simulated metrics
    ## Hospital incidence
    ## Calculate incidence and sample ihr and/or ihr multiplier
    ## ihr multiplier reduces severity for people with previous
    ## infection or vaccination

    ## Priors
    ### Intrinsic IHR
    with numpyro.plate("num_age", 4):
        ihr = numpyro.sample("ihr", dist.Beta(1, 9))

    ### IHR modified by previous infection or vaccination
    ihr_immune_mult = numpyro.sample("ihr_immune_mult", dist.Beta(4, 6))

    ### IHR modified by trends
    with numpyro.plate("K", K):
        ihr_coef = numpyro.sample(
            "ihr_coef", dist.TruncatedNormal(1.0, 0.5, low=0.2, high=3.0)
        )
    ihr_coeffs = jnp.append(jnp.array([1.0]), ihr_coef)

    ihr_mult_days = deBoor(
        jnp.searchsorted(ihr_knots, obs_days, "right") - 1,
        obs_days,
        ihr_knots,
        ihr_coeffs,
    )

    ## Calculations
    model_incidence = jnp.sum(sol.ys[3], axis=4)
    model_incidence_0 = jnp.diff(model_incidence[:, :, 0, 0], axis=0)

    model_incidence_1 = jnp.sum(model_incidence, axis=(2, 3))
    model_incidence_1 = jnp.diff(model_incidence_1, axis=0)
    model_incidence_1 -= model_incidence_0

    sim_incidence = (
        model_incidence_0 * ihr + model_incidence_1 * ihr * ihr_immune_mult
    )[obs_days] * ihr_mult_days[:, None]

    ## Seroprevalence
    never_infected = jnp.sum(sol.ys[0][sero_days, :, 0, :, :], axis=(2, 3))
    sim_seroprevalence = 1 - never_infected / m.config.POPULATION
    sim_lseroprevalence = jnp.log(
        sim_seroprevalence / (1 - sim_seroprevalence)
    )

    # Observation model
    numpyro.sample(
        "incidence",
        dist.Poisson(sim_incidence),
        obs=obs_incidence,
    )

    ## Masking nan observations due to incomplete seroprevalence data
    mask = ~jnp.isnan(obs_sero_lmean)
    with numpyro.handlers.mask(mask=mask):
        numpyro.sample(
            "lseroprevalence",
            dist.Normal(sim_lseroprevalence, obs_sero_lsd),
            obs=obs_sero_lmean,
        )
