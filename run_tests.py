import pytest
from seir_model_v4 import seirw_ode
from mechanistic_compartments import BasicMechanisticModel
import numpy as np

mechanistic_model = BasicMechanisticModel()
sol = mechanistic_model.run(seirw_ode, tf=1100.0, save_path=None)
times = sol.ts


def population_constant_test():
    summed_arrays = []
    # first we combine the age_groups, strain, and waning elements of each compartment
    for compartment in sol.ys:
        sum_across_time = np.array(compartment).sum(
            axis=tuple(range(1, compartment.ndim))
        )
        summed_arrays.append(sum_across_time)
    # then we combine all the compartments together to get total population at each timestep
    combined_compartments_across_time = np.round(sum(summed_arrays), decimals=3)
    print(combined_compartments_across_time)
    print(set(combined_compartments_across_time))

    assert (
        len(set(combined_compartments_across_time)) == 1
    )  # make sure summed values across all timesteps are equal


population_constant_test()
