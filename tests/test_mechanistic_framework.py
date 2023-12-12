from config.config_base import ConfigBase as mc_base
from config.config_r0_0 import ConfigScenario as mc_r0_0_vax_1in500
from config.config_r0_1_vax_0 import ConfigScenario as mc_r0_1_vax_0
from config.config_r0_15_vax_0_strain_2 import (
    ConfigScenario as config_r0_15_vax_0_strain_2,
)
from config.config_strain_2 import ConfigScenario as config_strain_2
from mechanistic_compartments import build_basic_mechanistic_model

model_r0_1_vax_0 = build_basic_mechanistic_model(mc_r0_1_vax_0())
model_r0_1_5_with_vax = build_basic_mechanistic_model(mc_base())
model_r0_0_with_vax = build_basic_mechanistic_model(mc_r0_0_vax_1in500())
model_r0_15_vax_0 = build_basic_mechanistic_model(
    config_r0_15_vax_0_strain_2()
)
model_strain_2 = build_basic_mechanistic_model(config_strain_2())
all_models = [
    model_r0_1_vax_0,
    model_r0_1_5_with_vax,
    model_r0_0_with_vax,
    model_r0_15_vax_0,
    model_strain_2,
]
