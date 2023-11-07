import numpy as np

from config.config_base import ConfigBase as config_base
from config.config_r0_0 import ConfigScenario as config_r0_0
from config.config_r0_1_vax_0 import ConfigScenario as config_r0_1_vax_0
from config.config_r0_15_vax_0_strain_2 import (
    ConfigScenario as config_r0_15_vax_0_strain_2,
)
from config.config_strain_2 import ConfigScenario as config_strain_2
from mechanistic_compartments import build_basic_mechanistic_model

# IMPORT MODEL YOU WISH TO TEST AND SET IT HERE
from model_odes.seir_model_v5 import seirw_ode, seirw_ode2

OG_MODEL = seirw_ode
MODEL = seirw_ode2

all_models = [
    build_basic_mechanistic_model(config_base()),
]


# TESTS BELOW
def test_model_equivalence():
    for test_model in all_models:
        args = test_model.get_args()
        state = test_model.INITIAL_STATE
        og_solution = OG_MODEL(state, 0, args)
        mo_solution = MODEL(state, 0, args)

        for i in range(len(og_solution)):
            assert np.allclose(
                og_solution[i], mo_solution[i], rtol=1e-5
            ), "The model solution is not identifical to OG model."
