import pytest

from mechanistic_model.covid_initializer import CovidInitializer

CONFIG_GLOBAL_PATH = "tests/test_config_global.json"
INITIALIZER_CONFIG_PATH = "tests/test_config_initializer.json"

initializer = CovidInitializer(INITIALIZER_CONFIG_PATH, CONFIG_GLOBAL_PATH)


def test_invalid_paths_raise():
    with pytest.raises(FileNotFoundError):
        CovidInitializer("random_broken_path", CONFIG_GLOBAL_PATH),
    with pytest.raises(FileNotFoundError):
        CovidInitializer(INITIALIZER_CONFIG_PATH, "random_broken_path")
    with pytest.raises(FileNotFoundError):
        CovidInitializer("random_broken_path", "random_broken_path2")


def test_initial_state_returns():
    assert hasattr(initializer, "INITIAL_STATE")
    assert initializer.get_initial_state() is not None


def test_num_compartments():
    assert (
        len(initializer.get_initial_state()) == 4
    ), "initial state returned not 4 compartments"


def test_initial_state_shape():
    expected_shape_S = (
        initializer.config.NUM_AGE_GROUPS,
        2**initializer.config.NUM_STRAINS,
        initializer.config.MAX_VACCINATION_COUNT + 1,
        initializer.config.NUM_WANING_COMPARTMENTS,
    )

    expected_shape_rest = (
        initializer.config.NUM_AGE_GROUPS,
        2**initializer.config.NUM_STRAINS,
        initializer.config.MAX_VACCINATION_COUNT + 1,
        initializer.config.NUM_STRAINS,
    )
    s, e, i, c = initializer.get_initial_state()
    assert (
        s.shape == expected_shape_S
    ), "Susceptible compartment shape incorrect"
    assert (
        e.shape == expected_shape_rest
    ), "Exposed compartment shape incorrect"
    assert (
        i.shape == expected_shape_rest
    ), "Infected compartment shape incorrect"
    assert (
        c.shape == expected_shape_rest
    ), "Book-keeping cumulative compartment shape incorrect"
