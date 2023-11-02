import jax.numpy as jnp
import numpy as np


class Parameters(object):
    """ A dummy container that converts a dictionary into attributes. """
    def __init__(self, dict: dict):
        self.__dict__ = dict


def seirw_ode(state, _, parameters):
    """
    A basic SEIRW ODE model to be used in solvers such as odeint or diffeqsolve

    Parameters:
    ----------
    state : array-like pytree
    a tuple or any array-like object capable of unpacking, holding the current state of the model,
    in this case holding population values of the S, E, I, R, and W compartments,

    _ : None
    Formally used to denote current time of the model, is not currently used in this function

    parameters : a dictionary
    a dictionary holding the values of parameters needed by the SEIRW model.

    Returns:
    a tuple containing the rates of change of all compartments given in the `state` parameter.
    each element in the return tuple will match the dimensions of the parallel element in `state`.

    """
    # Unpack state
    # dims s = (dc.NUM_AGE_GROUPS)
    # dims e/i/r = (dc.NUM_AGE_GROUPS, dc.NUM_STRAINS)
    # dims w = (dc.NUM_AGE_GROUPS, dc.NUM_STRAINS, mc.NUM_WANING_COMPARTMENTS)
    s, e, i, r, w = state
    p = Parameters(parameters)

    # TODO when adding birth and deaths just create it as a compartment
    force_of_infection = p.beta * p.contact_matrix.dot(i) / p.population[:, None]
    ds_to_e = force_of_infection * s[:, None]

    ds_to_w = s * p.vax_rate  # vaccination of suseptibles

    de_to_i = p.sigma * e  # exposure -> infectious
    di_to_r = p.gamma * i  # infectious -> recovered
    dr_to_w = p.waning_rate * r
    # guaranteed to wane into first waning compartment remaining in their strains.

    dw = jnp.zeros(w.shape)
    de = jnp.zeros(e.shape)
    # competition between strains for waned individuals + reinfection by same strain
    for strain_source_idx in range(p.num_strains):
        force_of_infection_strain = force_of_infection[:, strain_source_idx]
        for strain_target_idx in range(p.num_strains):
            # strain_source_idx will attempt to infect those previously infected with strain_target_idx.
            ws_by_age = w[:, strain_target_idx, :]
            partial_susceptibility = p.susceptibility_matrix[
                strain_source_idx, strain_target_idx
            ]
            effective_ws_by_age = ws_by_age * (
                1 - (p.waning_protections * (1 - partial_susceptibility))
            )
            ws_exposed = (
                force_of_infection_strain[:, None] * effective_ws_by_age
            )
            # element wise subtraction of exposed w_s from strain_target dw
            dw = dw.at[:, strain_target_idx, :].add(-ws_exposed)
            # element wise addition of exposed w_s into de
            de = de.at[:, strain_source_idx].add(np.sum(ws_exposed, axis=1))

    # lets measure our waned rates
    for w_idx in range(w.shape[-1]):
        # waning from waning compartment to waning compartment, last compartment does not wane
        w_waned = (
            0
            if w_idx == p.num_waning_compartments - 1
            else p.waning_rate * w[:, :, w_idx]
        )
        # waned individuals being vaccinated, no w1 -> w1 vaccination of top compartment
        w_vaxed = 0 if w_idx == 0 else p.vax_rate * w[:, :, w_idx]
        # persons gained from waning of compartments above
        # vaccination from all below compartments in the case of top compartment (to even out w_vaxed)
        w_gained = 0
        if w_idx == 0:
            w_gained = sum(
                [
                    p.vax_rate * w[:, :, w_idx_loop]
                    for w_idx_loop in range(w.shape[-1])
                    if w_idx_loop != 0
                    # we may want a dw1_to_w1 to represent recent infection getting vaccinated
                    # change w_vaxed in this case too
                ]
            )
        else:
            w_gained = p.waning_rate * w[:, :, w_idx - 1]

        dw = dw.at[:, :, w_idx].add(-w_waned)
        dw = dw.at[:, :, w_idx].add(w_gained)
        dw = dw.at[:, :, w_idx].add(-w_vaxed)
        if w_idx == 0:
            # waning from recovered into first waning compartment
            dw = dw.at[:, :, w_idx].add(dr_to_w)
            # vaccination from suseptible into first waning compartment, first strain only
            # TODO vaccination in alpha strain currently. move to omicron?
            dw = dw.at[:, 0, w_idx].add(ds_to_w)  # TODO fix this 0 hard code
        # last compartment doesnt wane
        # only top waning compartment receives people from "r"

    # sum ds_to_e since s does not split by subtype
    ds = jnp.add(
        jnp.zeros(s.shape), jnp.add(-jnp.sum(ds_to_e, axis=1), -ds_to_w)
    )
    de = jnp.add(de, -de_to_i + ds_to_e)

    di = jnp.add(jnp.zeros(i.shape), jnp.add(de_to_i, -di_to_r))
    dr = jnp.add(jnp.zeros(r.shape), jnp.add(di_to_r, -dr_to_w))
    return (ds, de, di, dr, dw)
