import jax.numpy as jnp
import pytest

from config.config import Config
from mechanistic_model.mechanistic_runner import MechanisticRunner
from model_odes.seip_model import seip_ode

CONFIG_GLOBAL_PATH = "tests/test_config_global.json"
RUNNER_CONFIG_PATH = "tests/test_config_runner.json"
global_config = Config(CONFIG_GLOBAL_PATH)
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
    500 * jnp.zeros(S_SHAPE),  # S
    jnp.zeros(EIC_SHAPE),  # E
    1 * jnp.zeros(EIC_SHAPE),  # I
    jnp.zeros(EIC_SHAPE),  # C
)
runner = MechanisticRunner(
    fake_initial_state, seip_ode, RUNNER_CONFIG_PATH, CONFIG_GLOBAL_PATH
)
ODES = seip_ode


def test_invalid_paths_raise():
    with pytest.raises(AssertionError):
        MechanisticRunner(
            fake_initial_state,
            seip_ode,
            "random_broken_path",
            CONFIG_GLOBAL_PATH,
        ),
    with pytest.raises(AssertionError):
        MechanisticRunner(
            fake_initial_state,
            seip_ode,
            RUNNER_CONFIG_PATH,
            "random_broken_path",
        )
    with pytest.raises(AssertionError):
        MechanisticRunner(
            fake_initial_state,
            seip_ode,
            "random_broken_path",
            "random_broken_path2",
        )


def test_external_i_shape():
    external_i_shape = runner.external_i(100).shape
    expected_shape = runner.INITIAL_STATE[runner.COMPARTMENT_IDX.I].shape
    assert (
        external_i_shape == expected_shape
    ), "external infections shape incompatible with I compartment"


def test_output_shapes():
    """tests that the ode-model outputs the correct compartment shapes according to the config file it was run in."""
    state = runner.INITIAL_STATE
    first_derivatives = ODES(state, 0, runner.get_args(sample=False))
    expected_output_shapes = [
        (
            runner.NUM_AGE_GROUPS,
            2**runner.NUM_STRAINS,
            runner.MAX_VAX_COUNT + 1,
            runner.NUM_WANING_COMPARTMENTS,
        ),
        (
            runner.NUM_AGE_GROUPS,
            2**runner.NUM_STRAINS,
            runner.MAX_VAX_COUNT + 1,
            runner.NUM_STRAINS,
        ),
        (
            runner.NUM_AGE_GROUPS,
            2**runner.NUM_STRAINS,
            runner.MAX_VAX_COUNT + 1,
            runner.NUM_STRAINS,
        ),
        (
            runner.NUM_AGE_GROUPS,
            2**runner.NUM_STRAINS,
            runner.MAX_VAX_COUNT + 1,
            runner.NUM_STRAINS,
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
