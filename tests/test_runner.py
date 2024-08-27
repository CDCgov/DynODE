import datetime

import jax.numpy as jnp
import pytest

from src import Config, MechanisticRunner, StaticValueParameters, utils
from src.model_odes.seip_model import seip_ode

CONFIG_GLOBAL_PATH = "tests/test_config_global.json"
RUNNER_CONFIG_PATH = "tests/test_config_runner.json"
GLOBAL_JSON = open(CONFIG_GLOBAL_PATH, "r").read()
global_config = Config(GLOBAL_JSON)
S_SHAPE = (
    global_config.NUM_AGE_GROUPS,
    2**global_config.NUM_STRAINS,
    global_config.MAX_VACCINATION_COUNT + 1,
    global_config.NUM_WANING_COMPARTMENTS,
)
EIC_SHAPE = (
    global_config.NUM_AGE_GROUPS,
    2**global_config.NUM_STRAINS,
    global_config.MAX_VACCINATION_COUNT + 1,
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
    external_i_shape = static_params.external_i(100, [], [], []).shape
    expected_shape = static_params.INITIAL_STATE[
        static_params.config.COMPARTMENT_IDX.I
    ].shape
    assert (
        external_i_shape == expected_shape
    ), "external infections shape incompatible with I compartment"


def test_scale_initial_infections():
    e = static_params.INITIAL_STATE[static_params.config.COMPARTMENT_IDX.E]
    i = static_params.INITIAL_STATE[static_params.config.COMPARTMENT_IDX.I]
    num_initial_infections = jnp.sum(e + i)
    # should always be zero since our fake state does not have exposures.
    # this is still fine to test because the ratios must sum to 1
    ratio_infections_exposed = jnp.sum(e) / num_initial_infections
    ratio_infections_infectious = jnp.sum(i) / num_initial_infections

    # test_config provides coverage for negative value test cases.
    # lets go through some common scale factors as well as the unchanged one and ensure that it remained the same
    for scale_factor in [0.5, 0.75, 1.0, 1.25, 1.5]:
        modified_initial_state = static_params.scale_initial_infections(
            scale_factor
        )
        e_modified = modified_initial_state[
            static_params.config.COMPARTMENT_IDX.E
        ]
        i_modified = modified_initial_state[
            static_params.config.COMPARTMENT_IDX.I
        ]
        num_initial_infections_modified = jnp.sum(e_modified + i_modified)
        ratio_infections_exposed_modified = (
            jnp.sum(e_modified) / num_initial_infections_modified
        )
        ratio_infections_infectious_modified = (
            jnp.sum(i_modified) / num_initial_infections_modified
        )
        # test that the number of infections actually increased by the correct number
        assert jnp.isclose(
            num_initial_infections * scale_factor,
            num_initial_infections_modified,
        ), (
            "scaling initial infections does not produce the correct number of new infections, "
            "began with %s infections, scaled by a factor of %d, ended with %s"
        ) % (
            str(num_initial_infections),
            scale_factor,
            str(num_initial_infections_modified),
        )
        # test that the E and I ratios are preserved.
        for ratio_start, ratio_end in zip(
            [ratio_infections_exposed, ratio_infections_infectious],
            [
                ratio_infections_exposed_modified,
                ratio_infections_infectious_modified,
            ],
        ):
            assert jnp.isclose(ratio_start, ratio_end), (
                "the ratio of infections into the exposed/infectious "
                "compartments changed after scaling initial infections. Started with ratio of %s ended with %s"
            ) % (str(ratio_start), str(ratio_end))


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
        static_params.config.VACCINATION_SEASON_CHANGE = season_change
        outflow_val = static_params.seasonal_vaccination_reset(
            utils.date_to_sim_day(
                season_change, static_params.config.INIT_DATE
            )
        )
        assert outflow_val == 1, (
            "seasonal outflow function does not peak on static_params.config.VACCINATION_SEASON_CHANGE like it should %s"
            % str(outflow_val)
        )


def test_seasonality_amplitude():
    static_params = StaticValueParameters(
        fake_initial_state,
        RUNNER_CONFIG_PATH,
        CONFIG_GLOBAL_PATH,
    )
    static_params.config.SEASONALITY_AMPLITUDE = 0.15
    seasonality_function = static_params.get_parameters()["SEASONALITY"]
    year_of_seasonality_curve = [seasonality_function(t) for t in range(365)]
    assert max(year_of_seasonality_curve) == 1.15


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
