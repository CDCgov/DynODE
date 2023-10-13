import pytest
from model_odes.seir_model_v4 import seirw_ode
from mechanistic_compartments import BasicMechanisticModel
from mechanistic_compartments import build_basic_mechanistic_model
import numpy as np
from config.config_base import ModelConfig as mc_base
from config.config_testing import ModelConfig as mc_test


mechanistic_model = build_basic_mechanistic_model(mc_base)
global_sol = mechanistic_model.run(seirw_ode, tf=1100.0, save_path=None)
global_times = global_sol.ts


def population_constant_test():
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


def r0_at_1_test():
    mechanistic_model = build_basic_mechanistic_model(mc_test)
    sol = mechanistic_model.run(seirw_ode, tf=1100.0, save_path=None)
    times = sol.ts
    suseptibles = sol.ys[mc_test.idx.S].sum(
        axis=tuple(range(1, sol.ys[mc_test.idx.S].ndim))
    )
    pearson_correlation_matrix = np.corrcoef(times, suseptibles)
    assert (
        np.round(np.abs(pearson_correlation_matrix), decimals=2) == 1
    ).all(), "R0 = 1.0 does not create linear decrease in suseptible population, implying issues with contact matrix normalizations"


population_constant_test()
r0_at_1_test()
