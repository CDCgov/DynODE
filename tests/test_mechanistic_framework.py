from mechanistic_compartments import build_basic_mechanistic_model
import numpy as np

from config.config_base import ModelConfig as mc_base
from config.config_r0_1_vax_0 import ModelConfig as mc_r0_1_vax_0
from config.config_r0_0_vax_1in500 import ModelConfig as mc_r0_0_vax_1in500
from config.config_r0_15_vax_0 import ModelConfig as mc_r0_15_vax_0

# from config.config_r0_15_vax_0_strain_2 import ModelConfig as mc_r0_15_vax_0_strain_2

model_r0_1_vax_0 = build_basic_mechanistic_model(mc_r0_1_vax_0)
model_r0_1_5_with_vax = build_basic_mechanistic_model(mc_base)
model_r0_0_with_vax = build_basic_mechanistic_model(mc_r0_0_vax_1in500)
model_r0_15_vax_0 = build_basic_mechanistic_model(mc_r0_15_vax_0)
# model_r0_15_vax_0_strain_2 = build_basic_mechanistic_model(mc_r0_15_vax_0_strain_2)
all_models = [
    model_r0_1_vax_0,
    model_r0_1_5_with_vax,
    model_r0_0_with_vax,
    model_r0_15_vax_0,
    # model_r0_15_vax_0_strain_2,
]


def test_uniform_initial_infection_distribution():
    """testing that the number of initial infections is uniformally distributed across strains"""
    for test_model in all_models:
        initial_infection_distribution = test_model.initial_state[test_model.idx.I]
        initial_infection_distribution = np.round(
            np.sum(initial_infection_distribution, axis=0), decimals=4
        )  # sum across age groups
        assert (
            len(set(initial_infection_distribution)) == 1
        ), "initial infections are not equally distributed across strains"
