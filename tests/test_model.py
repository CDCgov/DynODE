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
from model_odes.seir_model_v5 import seirw_ode

MODEL = seirw_ode

all_models = [
    build_basic_mechanistic_model(config_base()),
    build_basic_mechanistic_model(config_r0_0()),
    build_basic_mechanistic_model(config_r0_15_vax_0_strain_2()),
    build_basic_mechanistic_model(config_r0_1_vax_0()),
    build_basic_mechanistic_model(config_strain_2()),
]
no_vax_models = [
    build_basic_mechanistic_model(config_r0_15_vax_0_strain_2()),
    build_basic_mechanistic_model(config_r0_1_vax_0()),
]


# TESTS BELOW
def test_output_shapes():
    """tests that the ode-model outputs the correct compartment shapes according to the config file it was run in."""
    for test_model in all_models:
        state = test_model.INITIAL_STATE
        first_derivatives = MODEL(state, 0, test_model.get_args(sample=False))
        expected_output_shapes = [
            (test_model.NUM_AGE_GROUPS,),
            (test_model.NUM_AGE_GROUPS, test_model.NUM_STRAINS),
            (test_model.NUM_AGE_GROUPS, test_model.NUM_STRAINS),
            (test_model.NUM_AGE_GROUPS, test_model.NUM_STRAINS),
            (
                test_model.NUM_AGE_GROUPS,
                test_model.NUM_STRAINS,
                test_model.NUM_WANING_COMPARTMENTS,
            ),
        ]
        for compartment, expected_shape in zip(
            first_derivatives, expected_output_shapes
        ):
            assert compartment.shape == expected_shape, (
                "at least one compartment output shape does not match expected, was: "
                + str(compartment.shape)
                + " expected: "
                + str(expected_shape)
            )


def test_r0_at_1_constant_infections():
    """tests to make sure E + I derivative flows cancel out when
    R0 = 1, there are no vaccinations, and initial infections distributed based on contact matrix.

    TODO This test now fails because we initalize with non-uniform infections
    """
    model_r0_1_vax_0 = build_basic_mechanistic_model(config_r0_1_vax_0())
    state = model_r0_1_vax_0.INITIAL_STATE
    first_derivatives = MODEL(
        state, 0, model_r0_1_vax_0.get_args(sample=False)
    )
    (ds, de, di, dr, dw) = first_derivatives
    # sum across strains to just get age groups
    de = np.sum(de, axis=model_r0_1_vax_0.AXIS_IDX.strain)
    di = np.sum(di, axis=model_r0_1_vax_0.AXIS_IDX.strain)
    assert not np.round(de + di, decimals=4).any(), (
        "Inflow and outflow from de + di not canceling out when R0=1, "
        "there are no vaccinations, and initial infections distributed based on contact matrix"
    )


def test_constant_population():
    """tests that all models are not "creating" new people by making sure all derivatives sum out to zero"""
    for test_model in all_models:
        state = test_model.INITIAL_STATE
        first_derivatives = MODEL(state, 0, test_model.get_args(sample=False))
        (ds, de, di, dr, dw) = first_derivatives
        de = np.sum(de, axis=-1)  # sum across strains since ds has no strains
        di = np.sum(di, axis=-1)
        dr = np.sum(dr, axis=-1)
        dw = np.sum(
            dw, axis=(-1, -2)
        )  # also sum across waning compartments for waning
        assert not np.round(ds + de + di + dr + dw, decimals=4).any(), (
            "non-constant population across all compartments when creating a model with configurations: "
            + str(test_model.get_args())
        )


def test_no_exposed_r0_0():
    """tests that models with an r0 of 0 have no exposures, regardless of initial infections
    TODO this test now fails because we initalize with exposed persons in the INITIAL_STATE.
    """
    model_r0_0_with_vax = build_basic_mechanistic_model(config_r0_0())
    state = model_r0_0_with_vax.INITIAL_STATE
    first_derivatives = MODEL(
        state, 0, model_r0_0_with_vax.get_args(sample=False)
    )
    (ds, de, di, dr, dw) = first_derivatives
    de = np.sum(de, axis=-1)  # sum across strains to just get age groups
    assert not de.any(), "model still exposing new individuals even with r0=0"


def test_strains_equal():
    """tests that all models with equal r0s across strains and no vaccination
    have equal flows of people to each strain.
    Vaccination is excluded as persons are vaccinated for a particular strain

    TODO this test now fails because we initalize with all infected in the omicron strain, not uniform anymore
    """
    for test_model in no_vax_models:
        state = test_model.INITIAL_STATE
        first_derivatives = seirw_ode(
            state, 0, test_model.get_args(sample=False)
        )
        (ds, de, di, dr, dw) = first_derivatives
        dw = np.sum(dw, axis=-1)  # summing waning compartments
        for arr in [de, di, dr, dw]:
            cond = [
                all(arr[:, 0] == arr[:, x])
                for x in range(test_model.NUM_STRAINS)
            ]
            assert all(
                cond
            ), "equal r0s across strains + no vaccination should lead to strains having equal first derivatives"
