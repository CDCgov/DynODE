import pytest
from model_odes.seir_model_v4 import seirw_ode
from mechanistic_compartments import BasicMechanisticModel
from mechanistic_compartments import build_basic_mechanistic_model
import numpy as np
from config.config_base import ModelConfig as mc_base
from config.config_r0_1_vax_0 import ModelConfig as mc_test_r0_1_no_vax
import jax.numpy as jnp

MODEL = seirw_ode


mechanistic_model = build_basic_mechanistic_model(mc_test_r0_1_no_vax)
global_sol = mechanistic_model.run(MODEL, tf=1100.0, save_path=None)
global_times = global_sol.ts


def test_population_constant():
    summed_arrays = []
    # first we combine the age_groups, strain, and waning elements of each compartment
    for compartment in global_sol.ys:
        sum_across_time = np.array(compartment).sum(
            axis=tuple(range(1, compartment.ndim))
        )
        summed_arrays.append(sum_across_time)
    # then we combine all the compartments together to get total population at each timestep
    combined_compartments_across_time = np.round(sum(summed_arrays), decimals=6)

    assert (
        len(set(combined_compartments_across_time)) == 1
    )  # make sure summed values across all timesteps are equal


def test_r0_at_1_constant_infections():
    """ """
    state = mechanistic_model.initial_state
    first_derivatives = MODEL(state, 0, mechanistic_model.get_args(sample=False))
    (ds, de, di, dr, dw) = first_derivatives
    de = np.sum(de, axis=-1)  # sum across strains since ds has no strains
    di = np.sum(di, axis=-1)
    assert not (
        de + di
    ).any(), "Inflow and outflow from de + di not canceling out when R0=1"


def test_constant_population():
    state = mechanistic_model.initial_state
    first_derivatives = MODEL(state, 0, mechanistic_model.get_args(sample=False))
    (ds, de, di, dr, dw) = first_derivatives
    de = np.sum(de, axis=-1)  # sum across strains since ds has no strains
    di = np.sum(di, axis=-1)
    dr = np.sum(dr, axis=-1)
    dw = np.sum(dw, axis=(-1, -2))  # also sum across waning compartments for waning
    assert not (ds + de + di + dr + dw).any()


# population_constant_test()
# r0_at_1_test()
# r0_at_1_constant_infections_test()
# constant_population_test()
