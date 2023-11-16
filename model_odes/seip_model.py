import jax.numpy as jnp

# import numpy as np


class Parameters(object):
    """A dummy container that converts a dictionary into attributes."""

    def __init__(self, dict: dict):
        self.__dict__ = dict


def seip_ode(state, _, parameters):
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

    _ : None
    Formally used to denote current time of the model, is not currently used in this function

    parameters : a dictionary
    a dictionary holding the values of parameters needed by the SEIP model.

    Returns:
    a tuple containing the rates of change of all compartments given in the `state` parameter.
    each element in the return tuple will match the dimensions of the parallel element in `state`.

    """
    # strain dimension = 0-3 with some ENUM 0=no_prev_inf, 1 = prev_non_omicron, 2 = prev_omicron, 3=prev_both
    # s.shape = (NUM_AGE_GROUPS, strain, prev_vax_count, waned_state)
    # e/i/c .shape = (NUM_AGE_GROUPS, strain, prev_vax_count)
    # we dont have waning state once infection successful, waning state only impacts infection chances.
    s, e, i, c = state
    p = Parameters(parameters)
    ds, de = jnp.zeros(s.shape), jnp.zeros(e.shape)
    de = de  # to pass precommit
    # shape of (num_age_groups, num_strains)
    # TODO how does this work?
    for strain_source_idx in range(p.num_strains):
        # force_of_infection_strain = force_of_infection[:, strain_source_idx]
        # strain_source_idx will attempt to infect those previously infected with strain_target_idx.
        # susceptibility_matrix.shape = (num_strains, 2^num_strains) matrix of strain vs exposure hist.
        # partial_susceptibility_by_strain.shape = (2^num_strains,)
        partial_susceptibility_by_strain = p.susceptibility_matrix[
            strain_source_idx
        ]
        # partial_vax_susceptiblity_by_strain.shape = (max_vax_count+1.,)
        partial_vax_susceptiblity_by_strain = p.vax_eff_matrix[
            strain_source_idx
        ]
        immune_state_susceptibility = jnp.outer(
            partial_susceptibility_by_strain,
            partial_vax_susceptiblity_by_strain,
        )
        effective_exposed = (
            immune_state_susceptibility * p.waning_protections
        )  # something here?
        effective_exposed = effective_exposed  # passing precommit
        # dw = dw.at[:, strain_target_idx, :].add(-ws_exposed)
        # de = de.at[:, strain_source_idx].add(np.sum(ws_exposed, axis=1))

    # e and i shape remain same, just multiplying by a constant.
    de_to_i = p.sigma * e  # exposure -> infectious
    # s[:, a, b, c] -> s[:, d, b, 0], if a = both_prev_inf (4), then a=d. immune event just ended, zeroth waning compartment.
    di_to_w0 = p.gamma * i  # infectious -> new_immune_state
    di = jnp.add(de_to_i, -di_to_w0)

    # lets measure our waned + vax rates
    # TODO, change waning_rates because we no longer have R->W
    # last w group doesn't wane but waning_rates enforces a 0 at the end
    waning_array = jnp.zeros(s.shape).at[:, :, :].add(p.waning_rates)
    s_waned = waning_array * s
    ds.at[:, :, :, 1 : p.num_waning].add(s_waned[:, :, 0 : p.num_waning - 1])
    # TODO here we need to add our di_to_w0 but in a smarter way to sort out prev_exposure column
    # ds.at[:, :, :, 0]

    # slice across age, strain, and wane. vaccination updates the vax column and also moves all to w0.
    # ex: diagonal movement from 1 shot in 4th waning compartment to 2 shots 0 waning compartment      s[:, 0, 1, 3] -> s[:, 0, 2, 0]
    for vaccine_count in range(p.max_vax_count + 1):
        # num of people who had vaccine_count shots and then are getting 1 more
        s_vax_count = p.vax_rate * s[:, :, vaccine_count, :]
        # TODO, do people in W0 get vaccinated? They just recovered so I am going with no.
        # this also means people who just got vaccinated wont get another one for at least 1 waning compartment time.
        s_vax_count.at[:, :, vaccine_count, 0].set(0)
        # sum all the people getting vaccines, since they will be put in w0
        vax_gained = sum(s_vax_count, axis=-1)
        # if people already at the max counted vaccinations, dont move them, only update waning
        if vaccine_count == p.max_vax_count:
            ds.at[:, :, vaccine_count, 0].add(vax_gained)
        else:  # increment num_vaccines by 1, waning reset
            ds.at[:, :, vaccine_count + 1, 0].add(vax_gained)
        # set the w0 compartment to 0 here since we are going to subtract it from the other waning compartments
        ds.at[:, :, vaccine_count, :].add(-s_vax_count)

    return (ds, de, di, jnp.zeros(c.shape))
