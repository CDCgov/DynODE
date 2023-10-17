from mechanistic_compartments import build_basic_mechanistic_model
import numpy as np

# IMPORT MODEL YOU WISH TO TEST AND SET IT HERE
from model_odes.seir_model_v4 import seirw_ode

MODEL = seirw_ode
# create mechanistic model wrapper to load contact matrix and params from each config file
from config.config_base import ModelConfig as mc_base
from config.config_r0_1_vax_0 import ModelConfig as mc_r0_1_no_vax
from config.config_r0_0_vax_1in500 import ModelConfig as mc_r0_0_vax_1in500

model_r0_1_no_vax = build_basic_mechanistic_model(mc_r0_1_no_vax)
model_r0_1_5_with_vax = build_basic_mechanistic_model(mc_base)
model_r0_0_with_vax = build_basic_mechanistic_model(mc_r0_0_vax_1in500)
all_models = [model_r0_1_no_vax, model_r0_1_5_with_vax, model_r0_0_with_vax]


# TESTS BELOW
def test_r0_at_1_constant_infections():
    """tests to make sure E + I derivative flows cancel out when
    R0 = 1, there are no vaccinations, and initial infections distributed based on contact matrix
    """
    state = model_r0_1_no_vax.initial_state
    first_derivatives = MODEL(state, 0, model_r0_1_no_vax.get_args(sample=False))
    (ds, de, di, dr, dw) = first_derivatives
    de = np.sum(de, axis=-1)  # sum across strains to just get age groups
    di = np.sum(di, axis=-1)
    assert not np.round(de + di, decimals=4).any(), (
        "Inflow and outflow from de + di not canceling out when R0=1, "
        "there are no vaccinations, and initial infections distributed based on contact matrix"
    )


def test_constant_population():
    """tests that all models are not "creating" new people by making sure all derivatives sum out to zero"""
    for test_model in all_models:
        state = test_model.initial_state
        first_derivatives = MODEL(state, 0, test_model.get_args(sample=False))
        (ds, de, di, dr, dw) = first_derivatives
        de = np.sum(de, axis=-1)  # sum across strains since ds has no strains
        di = np.sum(di, axis=-1)
        dr = np.sum(dr, axis=-1)
        dw = np.sum(dw, axis=(-1, -2))  # also sum across waning compartments for waning
        assert not np.round(ds + de + di + dr + dw, decimals=4).any(), (
            "non-constant population across all compartments when creating a model with configurations: "
            + str(test_model.get_args())
        )


def test_no_exposed_r0_0():
    """tests that models with an r0 of 0 have no exposures, regardless of initial infections"""
    state = model_r0_0_with_vax.initial_state
    first_derivatives = MODEL(state, 0, model_r0_0_with_vax.get_args(sample=False))
    (ds, de, di, dr, dw) = first_derivatives
    de = np.sum(de, axis=-1)  # sum across strains to just get age groups
    assert not de.any(), "model still exposing new individuals even with r0=0"
