import jax.numpy as jnp
import pytest

from config.config import Config
from mechanistic_model.mechanistic_runner import MechanisticRunner
from mechanistic_model.static_value_parameters import StaticValueParameters
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
    external_i_shape = static_params.external_i(100).shape
    expected_shape = static_params.INITIAL_STATE[
        static_params.config.COMPARTMENT_IDX.I
    ].shape
    assert (
        external_i_shape == expected_shape
    ), "external infections shape incompatible with I compartment"


def test_output_shapes():
    """tests that the ode-model outputs the correct compartment shapes according to the config file it was run in."""
    first_derivatives = ODES(
        static_params.INITIAL_STATE, 0, static_params.get_parameters()
    )
    expected_output_shapes = [
        (
            static_params.config.NUM_AGE_GROUPS,
            2**static_params.config.NUM_STRAINS,
            static_params.config.MAX_VAX_COUNT + 1,
            static_params.config.NUM_WANING_COMPARTMENTS,
        ),
        (
            static_params.config.NUM_AGE_GROUPS,
            2**static_params.config.NUM_STRAINS,
            static_params.config.MAX_VAX_COUNT + 1,
            static_params.config.NUM_STRAINS,
        ),
        (
            static_params.config.NUM_AGE_GROUPS,
            2**static_params.config.NUM_STRAINS,
            static_params.config.MAX_VAX_COUNT + 1,
            static_params.config.NUM_STRAINS,
        ),
        (
            static_params.config.NUM_AGE_GROUPS,
            2**static_params.config.NUM_STRAINS,
            static_params.config.MAX_VAX_COUNT + 1,
            static_params.config.NUM_STRAINS,
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
