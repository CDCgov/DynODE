import datetime

import jax.numpy as jnp
import pytest

import utils
from config.config import Config
from mechanistic_model.mechanistic_runner import MechanisticRunner
from mechanistic_model.static_value_parameters import StaticValueParameters
from model_odes.seip_model import seip_ode

CONFIG_GLOBAL_PATH = "tests/test_config_global.json"
RUNNER_CONFIG_PATH = "tests/test_config_runner.json"
GLOBAL_JSON = open(CONFIG_GLOBAL_PATH, "r").read()
global_config = Config(GLOBAL_JSON)
S_SHAPE = (
    global_config.NUM_AGE_GROUPS,
    2**global_config.NUM_STRAINS,
    global_config.MAX_VAX_COUNT + 1,
    global_config.NUM_WANING_COMPARTMENTS,
)
EIC_SHAPE = (
    global_config.NUM_AGE_GROUPS,
    2**global_config.NUM_STRAINS,
    global_config.MAX_VAX_COUNT + 1,
    global_config.NUM_STRAINS,
)
fake_initial_state = (
    500 * jnp.ones(S_SHAPE),  # S
    jnp.zeros(EIC_SHAPE),  # E
    1 * jnp.ones(EIC_SHAPE),  # I
    jnp.zeros(EIC_SHAPE),  # C
)
static_params = StaticValueParameters(
    fake_initial_state,
    RUNNER_CONFIG_PATH,
    CONFIG_GLOBAL_PATH,
)
runner = MechanisticRunner(seip_ode)
ODES = seip_ode


def test_invalid_paths_raise():
    with pytest.raises(FileNotFoundError):
        StaticValueParameters(
            fake_initial_state,
            "random_broken_path",
            CONFIG_GLOBAL_PATH,
        ),
    with pytest.raises(FileNotFoundError):
        StaticValueParameters(
            fake_initial_state,
            RUNNER_CONFIG_PATH,
            "random_broken_path",
        )
    with pytest.raises(FileNotFoundError):
        StaticValueParameters(
            fake_initial_state,
            "random_broken_path",
            "random_broken_path2",
        )


def test_external_i_shape():
    external_i_shape = static_params.external_i(100, []).shape
    expected_shape = static_params.INITIAL_STATE[
        static_params.config.COMPARTMENT_IDX.I
    ].shape
    assert (
        external_i_shape == expected_shape
    ), "external infections shape incompatible with I compartment"


def test_seasonal_vaccination_reset():
    static_params = StaticValueParameters(
        fake_initial_state,
        RUNNER_CONFIG_PATH,
        CONFIG_GLOBAL_PATH,
    )
    static_params.config.SEASONAL_VACCINATION = True
    # set a number of season change points and test the function for each one
    for month in range(1, 13):
        season_change = static_params.config.INIT_DATE + datetime.timedelta(
            days=30 * month
        )
        static_params.config.VAX_SEASON_CHANGE = season_change
        outflow_val = static_params.seasonal_vaccination_reset(
            utils.date_to_sim_day(
                season_change, static_params.config.INIT_DATE
            )
        )
        assert outflow_val == 1, (
            "seasonal outflow function does not peak on static_params.config.VAX_SEASON_CHANGE like it should %s"
            % str(outflow_val)
        )


def test_output_shapes():
    """tests that the ode-model outputs the correct compartment shapes according to the config file it was run in."""
    first_derivatives = ODES(
        static_params.INITIAL_STATE, 0, static_params.get_parameters()
    )
    expected_output_shapes = [
        S_SHAPE,
        EIC_SHAPE,
        EIC_SHAPE,
        EIC_SHAPE,
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


def test_non_negative_compartments():
    solution = runner.run(fake_initial_state, static_params.get_parameters())
    for compartment_name, compartment in zip(
        static_params.config.COMPARTMENT_IDX._member_names_, solution.ys
    ):
        assert (compartment >= 0).all(), (
            "compartment counts must be above zero, found negatives in compartment %s"
            % compartment_name
        )
