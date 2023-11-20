import jax.numpy as jnp
import numpy as np

from utils import new_immune_state


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
    # s.shape = (NUM_AGE_GROUPS, hist, prev_vax_count, waned_state)
    # e/i/c .shape = (NUM_AGE_GROUPS, hist, prev_vax_count, strain)
    # we dont have waning state once infection successful, waning state only impacts infection chances.
    s, e, i, c = state
    p = Parameters(parameters)
    ds, de, di, dc = (
        jnp.zeros(s.shape),
        jnp.zeros(e.shape),
        jnp.zeros(i.shape),
        jnp.zeros(c.shape),
    )
    # CALCULATING SUCCESSFULL INFECTIONS OF (partially) SUSCEPTIBLE INDIVIDUALS
    force_of_infection = (
        (p.BETA * np.einsum("ab,bijk->aijk", p.CONTACT_MATRIX, i)).transpose()
        / p.POPULATION
    ).transpose()  # (NUM_AGE_GROUPS, hist, prev_vax_count, strain)

    for strain in range(p.NUM_STRAINS):
        force_of_infection_strain = force_of_infection[
            :, :, :, strain
        ]  # (num_age_groups, hist, max_vax_count)

        # partial_suseptibility = (hist,)
        partial_susceptibility = p.SUSCEPTIBILITY_MATRIX[strain, :]
        # p.vax_susceptibility_strain.shape = (max_vax_count,)
        vax_susceptibility_strain = p.VAX_EFF_MATRIX[strain, :]

        effective_susceptibility = 1 - jnp.matmul(
            p.WANING_PROTECTIONS[:, None],
            (1 - partial_susceptibility)[None, :],
        )  # (waning, immune_state,)
        exposed_s = jnp.array(
            [
                np.einsum(
                    "b,abc->abc",
                    effective_susceptibility[wane, :],
                    (
                        force_of_infection_strain
                        * s[:, :, :, wane]
                        * vax_susceptibility_strain[None, :]
                    ),
                )
                for wane in range(s.shape[-1])
            ]
        )
        # equation: (immune_state) x ((num_age_groups, immune_state, max_vax_count)**2 * (1, max_vax_count))
        # for loop prepends a (wane) dimension to this array, tranpose into correct order
        exposed_s = exposed_s.transpose((1, 2, 3, 0))
        # we know strain_source is infecting people, de has no waning compartments, so sum over those.
        de = de.at[:, :, :, strain].add(jnp.sum(exposed_s, axis=-1))
        ds = ds.add(-exposed_s)
        # de = de.at[:, strain_source_idx].add(np.sum(ws_exposed, axis=1))

    # e and i shape remain same, just multiplying by a constant.
    de_to_i = p.sigma * e  # exposure -> infectious
    di_to_w0 = p.gamma * i  # infectious -> new_immune_state
    di = jnp.add(de_to_i, -di_to_w0)
    de = de.add(-de_to_i)

    for strain, immune_state in zip(range(p.NUM_STRAINS), s.shape[-1]):
        new_state = new_immune_state(immune_state, strain, p.NUM_STRAINS)
        # recovered i->w0 transfer from `immune_state` -> `new_state` due to recovery from `strain`
        ds = ds.at[:, new_state, :, 0].add(
            di_to_w0.at[:, immune_state, :, strain]
        )
        # TODO this is where some percentage of the recovery goes to death or hosptialization

    # lets measure our waned + vax rates
    # TODO, change waning_rates because we no longer have R->W
    # last w group doesn't wane but waning_rates enforces a 0 at the end
    waning_array = jnp.zeros(s.shape).at[:, :, :].add(p.waning_rates)
    s_waned = waning_array * s
    ds.at[:, :, :, 1 : p.num_waning].add(s_waned[:, :, 0 : p.num_waning - 1])
    # TODO forgot to subtract the waning only added people here
    # TODO here we need to add our di_to_w0 but in a smarter way to sort out prev_exposure column
    # with num_strains=2 we get 2^2 immune_states. no exposure, strain 0 only, strain 1 only, both strains.

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

    return (ds, de, di, dc)
