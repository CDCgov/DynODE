from itertools import product

import jax.numpy as jnp

from utils import Parameters, get_foi_suscept, new_immune_state


def seip_ode(state, t, parameters):
    """
    A immune state ode model which aims to represent an SEIP model, with better representation of immune state and partial suseptibility.

    In previous version of the model those in the suseptible compartment were considered fully suseptible,
    and those with prior immune exposure would be placed in other compartments and wane in those compartments.

    This ODE model aims to represent suseptible people by a complex immune state,
    fully suseptible people are included in the suseptible compartment but are a smaller subset of it.

    the S compartment is now stratified by the following dimensions:
    age, previous omicron infection, previous non-omicron infection, number of vaccinations, and most recent immune event waning.

    A individual who is 15, with 1 previous omicron infection and 2 vaccinations will be marked as such,
    with the *more* recent between their vaccination and their omicron infection placing them in a waning compartment.


    Parameters:
    ----------
    state : array-like pytree
    a tuple or any array-like object capable of unpacking, holding the current state of the model,
    in this case holding population values of the S, E, I, and C compartments. (note: C =
    cumulative incidence).

    t : int
    used to denote current time of the model in days

    parameters : a dictionary
    a dictionary holding the values of parameters needed by the SEIP model.

    Returns:
    a tuple containing the rates of change of all compartments given in the `state` parameter.
    each element in the return tuple will match the dimensions of the parallel element in `state`.

    """
    # strain dimension = 0-3 with some ENUM 0=no_prev_inf, 1 = prev_non_omicron, 2 = prev_omicron, 3=prev_both
    # s.shape = (NUM_AGE_GROUPS, hist, prev_vax_count, waned_state)
    # e/i/c .shape = (NUM_AGE_GROUPS, hist, prev_vax_count, strain)
    # we dont have waning state once infection successful, waning state only impacts infection chances.
    s, e, i, c = state
    # spoof the dict into a class so we can use `p.` notation instead of dicts
    p = Parameters(parameters)
    ds, de, di, dc = (
        jnp.zeros(s.shape),
        jnp.zeros(e.shape),
        jnp.zeros(i.shape),
        jnp.zeros(c.shape),
    )
    beta_coef = p.BETA_COEF(t)
    seasonality_coef = p.SEASONALITY(t)
    # CALCULATING SUCCESSFULL INFECTIONS OF (partially) SUSCEPTIBLE INDIVIDUALS
    # including externally infected individuals to introduce new strains
    force_of_infection = (
        (
            p.BETA
            * beta_coef
            * seasonality_coef
            * jnp.einsum(
                "ab,bijk->ak",
                p.CONTACT_MATRIX,
                i + p.EXTERNAL_I(t),
            )
        ).transpose()
        / p.POPULATION
    ).transpose()  # (NUM_AGE_GROUPS, strain)

    foi_suscept = get_foi_suscept(p, force_of_infection)
    for strain in range(p.NUM_STRAINS):
        exposed_s = s * foi_suscept[strain]
        # we know `strain` is infecting people, de has no waning compartments, so sum over those.
        de = de.at[:, :, :, strain].add(jnp.sum(exposed_s, axis=-1))
        ds = jnp.add(ds, -exposed_s)
    dc = de  # at this point we only have infections in de, so we add to cumulative
    # e and i shape remain same, just multiplying by a constant.
    de_to_i = p.SIGMA * e  # exposure -> infectious
    di_to_w0 = p.GAMMA * i  # infectious -> new_immune_state
    di = jnp.add(de_to_i, -di_to_w0)
    de = jnp.add(de, -de_to_i)

    # go through all combinations of immune history and exposing strain
    # calculate new immune history after recovery, place them there.
    for strain, immune_state in product(
        range(p.NUM_STRAINS), range(2**p.NUM_STRAINS)
    ):
        new_state = new_immune_state(immune_state, strain, p.NUM_STRAINS)
        # recovered i->w0 transfer from `immune_state` -> `new_state` due to recovery from `strain`
        ds = ds.at[:, new_state, :, 0].add(
            di_to_w0[:, immune_state, :, strain]
        )
        # TODO this is where some percentage of the recovery goes to death or hosptialization

    # lets measure our waned + vax rates
    # last w group doesn't wane but WANING_RATES enforces a 0 at the end
    waning_array = jnp.zeros(s.shape).at[:, :, :].add(p.WANING_RATES)
    s_waned = waning_array * s
    ds = ds.at[:, :, :, 1:].add(s_waned[:, :, :, :-1])
    ds = ds.at[:, :, :, :-1].add(-s_waned[:, :, :, :-1])

    # slice across age, strain, and wane. vaccination updates the vax column and also moves all to w0.
    # ex: diagonal movement from 1 shot in 4th waning compartment to 2 shots 0 waning compartment      s[:, 0, 1, 3] -> s[:, 0, 2, 0]
    # input vaccination rate is per entire population, need to update to per compartments first
    vax_rates = p.VACCINATION_RATES(t)
    vax_totals = vax_rates * p.POPULATION[:, None]
    vax_status_counts = jnp.sum(
        s, axis=(1, 3)
    )  # Sum over immune hist and waning to get count per age and vax status
    updated_vax_rates = vax_totals / vax_status_counts
    updated_vax_rates = jnp.where(
        updated_vax_rates > 1.0,
        jnp.ones(updated_vax_rates.shape),
        updated_vax_rates,
    )  # prevent moving more people out than the compartments have

    # Assuming that people who received 2 or more doses wouldn't get additional booster too soon
    # i.e., when they were still within the first waning compartment
    vax_counts = s * updated_vax_rates[:, jnp.newaxis, :, jnp.newaxis]
    vax_counts = vax_counts.at[:, :, p.MAX_VAX_COUNT, 0].set(0)
    vax_gained = jnp.sum(vax_counts, axis=(-1))
    ds = ds.at[:, :, p.MAX_VAX_COUNT, 0].add(vax_gained[:, :, p.MAX_VAX_COUNT])
    ds = ds.at[:, :, 1 : (p.MAX_VAX_COUNT) + 1, 0].add(
        vax_gained[:, :, 0 : p.MAX_VAX_COUNT]
    )
    ds = ds - vax_counts

    # if we are not implementing seasonal vaccination p.SEASONAL_VACCINATION_RESET(t) = 0 forall t
    # and you can safely ignore this section
    seasonal_vaccination_outflow = p.SEASONAL_VACCINATION_RESET(t)
    # flow seasonal_vaccination_outflow% of seasonal vaxers back to max ordinal tier
    ds = ds.at[:, :, p.MAX_VAX_COUNT - 1, :].add(
        seasonal_vaccination_outflow * s[:, :, p.MAX_VAX_COUNT, :]
    )
    # remove these people from the seasonal vaccination tier
    ds = ds.at[:, :, p.MAX_VAX_COUNT, :].add(
        -seasonal_vaccination_outflow * s[:, :, p.MAX_VAX_COUNT, :]
    )
    # do the same process for e and i compartments
    de = de.at[:, :, p.MAX_VAX_COUNT - 1, :].add(
        seasonal_vaccination_outflow * e[:, :, p.MAX_VAX_COUNT, :]
    )
    de = de.at[:, :, p.MAX_VAX_COUNT, :].add(
        -seasonal_vaccination_outflow * e[:, :, p.MAX_VAX_COUNT, :]
    )
    di = di.at[:, :, p.MAX_VAX_COUNT - 1, :].add(
        seasonal_vaccination_outflow * i[:, :, p.MAX_VAX_COUNT, :]
    )
    di = di.at[:, :, p.MAX_VAX_COUNT, :].add(
        -seasonal_vaccination_outflow * i[:, :, p.MAX_VAX_COUNT, :]
    )

    return (ds, de, di, dc)
