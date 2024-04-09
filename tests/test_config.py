"""
Testing the Config parser, downstream parameters, errors, and appropriate fail safes
"""

import json

import numpyro.distributions as dist
import pytest

from config.config import Config, ConfigParserError

GLOBAL_TEST_CONFIG = "tests/test_config_global.json"
PATH_VARIABLES = [
    "SAVE_PATH",
    "DEMOGRAPHIC_DATA_PATH",
    "SEROLOGICAL_DATA_PATH",
    "SIM_DATA_PATH",
    "VAX_MODEL_DATA",
]


def test_invalid_json():
    # missing closing bracket
    example_invalid_json_input = """{ "INFECTIOUS_PERIOD": 5"""
    with pytest.raises(json.decoder.JSONDecodeError):
        Config(example_invalid_json_input)


def test_valid_path_variables():
    # testing valid paths
    for path_var in PATH_VARIABLES:
        example_input_json = """{"%s":"%s"}""" % (path_var, GLOBAL_TEST_CONFIG)
        assert (
            Config(example_input_json).__dict__[path_var] == GLOBAL_TEST_CONFIG
        )


def test_invalid_type_path_variables():
    for path_var in PATH_VARIABLES:
        example_input_json = """{"%s":%d}""" % (path_var, 10)
        with pytest.raises(AssertionError):
            Config(example_input_json)
    for path_var in PATH_VARIABLES:
        example_input_json = """{"%s":"%s"}""" % (
            path_var,
            "some_random_incorrect_path.json",
        )
        print(example_input_json)
        with pytest.raises(AssertionError):
            Config(example_input_json)


def test_non_ascending_age_limits():
    input_json = """{"AGE_LIMITS": [10, 1, 50, 60]}"""
    with pytest.raises(AssertionError):
        Config(input_json)


def test_out_of_bounds_age_limits():
    input_json = """{"AGE_LIMITS": [0, 18, 50, 64, 95]}"""
    with pytest.raises(AssertionError):
        Config(input_json)


def test_negative_age_limits():
    input_json = """{"AGE_LIMITS": [-10, 18, 50, 64]}"""
    with pytest.raises(AssertionError):
        Config(input_json)


def test_float_ages():
    input_json = """{"AGE_LIMITS": [0, 5.5, 18, 50, 64]}"""
    with pytest.raises(AssertionError):
        Config(input_json)


def test_valid_age_limits():
    input_json = """{"AGE_LIMITS": [0, 5, 18, 50, 64]}"""
    assert Config(input_json).AGE_LIMITS == [0, 5, 18, 50, 64]


def test_negative_population_size():
    input_json = """{"POP_SIZE": -1}"""
    with pytest.raises(AssertionError):
        Config(input_json)


def test_str_population_size():
    input_json = """{"POP_SIZE": "5"}"""
    with pytest.raises(AssertionError):
        Config(input_json)


def test_valid_population_size():
    input_json = """{"POP_SIZE": 100}"""
    assert Config(input_json).POP_SIZE == 100


def test_negative_initial_infections():
    input_json = """{"INITIAL_INFECTIONS": -5}"""
    with pytest.raises(AssertionError):
        Config(input_json)


def test_negative_tree_depth():
    input_json = """{"MAX_TREE_DEPTH": -1}"""
    with pytest.raises(AssertionError):
        Config(input_json)


def test_zero_tree_depth():
    input_json = """{"MAX_TREE_DEPTH": 0}"""
    with pytest.raises(AssertionError):
        Config(input_json)


def test_float_tree_depth():
    input_json = """{"MAX_TREE_DEPTH": 1.2}"""
    with pytest.raises(AssertionError):
        Config(input_json)


def test_valid_tree_depth():
    input_json = """{"MAX_TREE_DEPTH": 10}"""
    assert Config(input_json).MAX_TREE_DEPTH == 10


def test_str_initial_infections():
    input_json = """{"INITIAL_INFECTIONS": "5"}"""
    with pytest.raises(AssertionError):
        Config(input_json)


def test_valid_initial_infections():
    input_json = """{"INITIAL_INFECTIONS": 5}"""
    assert Config(input_json).INITIAL_INFECTIONS == 5


def test_valid_initial_infections_float():
    input_json = """{"INITIAL_INFECTIONS": 5.3}"""
    assert Config(input_json).INITIAL_INFECTIONS == 5.3


def test_init_infections_greater_than_pop_size():
    input_json = """{"POP_SIZE": 1, "INITIAL_INFECTIONS": 5}"""
    with pytest.raises(AssertionError):
        Config(input_json)


def test_init_infections_less_than_pop_size():
    input_json = """{"POP_SIZE": 5, "INITIAL_INFECTIONS": 1}"""
    c = Config(input_json)
    assert c.POP_SIZE == 5 and c.INITIAL_INFECTIONS == 1


def test_valid_infectious_period():
    input_json = """{"INFECTIOUS_PERIOD": 5}"""
    c = Config(input_json)
    assert c.INFECTIOUS_PERIOD == 5


def test_negative_infectious_period():
    input_json = """{"INFECTIOUS_PERIOD": -5}"""
    with pytest.raises(AssertionError):
        Config(input_json)


def test_invalid_support_infectious_period():
    input_json = """{"INFECTIOUS_PERIOD": {"distribution": "Normal", "params": {"loc": 0, "scale":1}}}"""
    with pytest.raises(AssertionError):
        Config(input_json)


def test_valid_support_infectious_period():
    input_json = """{"INFECTIOUS_PERIOD": {"distribution": "TruncatedNormal", "params": {"loc": 2, "scale":1, "low":1}}}"""
    c = Config(input_json)
    assert issubclass(type(c.INFECTIOUS_PERIOD), dist.Distribution)


def test_valid_nested_distribution_infectious_period():
    # a basic 1 + N(0,1) distribution
    input_json = """{"INFECTIOUS_PERIOD": {
            "distribution": "TransformedDistribution",
            "params": {
                "base_distribution": {
                    "distribution": "TruncatedNormal",
                    "params": {
                        "loc": 5,
                        "scale": 1,
                        "low": 1
                    }
                },
                "transforms": {
                    "transform": "AffineTransform",
                    "params": {
                        "loc": 1,
                        "scale": 1,
                        "domain": {
                            "constraint": "greater_than",
                            "params": {
                                "lower_bound": 1
                            }
                        }
                    }
                }
            }
        }}"""
    c = Config(input_json)
    assert isinstance(c.INFECTIOUS_PERIOD, dist.TransformedDistribution)


def test_invalid_distribution_infectious_period():
    input_json = """{"INFECTIOUS_PERIOD": {"distribution": "Normal", "params": {"loc": 0, "scale":"blah"}}}"""
    with pytest.raises(ConfigParserError):
        Config(input_json)


def test_invalid_params_nested_distribution_infectious_period():
    # a basic 1 + N(0,1) distribution
    input_json = """{"INFECTIOUS_PERIOD": {
            "distribution": "TransformedDistribution",
            "params": {
                "base_distribution": {
                    "distribution": "TruncatedNormal",
                    "params": {
                        "loc": 5,
                        "scale": 1,
                        "low": 1
                    }
                },
                "transforms": {
                    "transform": "AffineTransform",
                    "params": {
                        "loc": "invalid_input",
                        "scale": 1,
                        "domain": {
                            "constraint": "greater_than",
                            "params": {
                                "lower_bound": 1
                            }
                        }
                    }
                }
            }
        }}"""
    with pytest.raises(ConfigParserError):
        Config(input_json)


def test_invalid_support_nested_distribution_infectious_period():
    # a basic 1 + N(0,1) distribution
    input_json = """{"INFECTIOUS_PERIOD": {
            "distribution": "TransformedDistribution",
            "params": {
                "base_distribution": {
                    "distribution": "Normal",
                    "params": {
                        "loc": 0,
                        "scale": 1
                    }
                },
                "transforms": {
                    "transform": "AffineTransform",
                    "params": {
                        "loc": 1,
                        "scale": 1
                    }
                }
            }
        }}"""
    with pytest.raises(AssertionError):
        Config(input_json)


def test_invalid_constraint_param():
    # a basic 1 + N(0,1) distribution with incorrect interval constraint
    input_json = """{"INFECTIOUS_PERIOD": {
            "distribution": "TransformedDistribution",
            "params": {
                "base_distribution": {
                    "distribution": "Normal",
                    "params": {
                        "loc": 0,
                        "scale": 1
                    }
                },
                "transforms": {
                    "transform": "AffineTransform",
                    "params": {
                        "loc": 1,
                        "scale": 1,
                        "domain": {
                            "constraint": "interval",
                            "params": {
                                "lower_bound": 0,
                                "upper_bound": "blah"
                            }
                        }
                    }
                }
            }
        }}"""
    with pytest.raises(ConfigParserError):
        Config(input_json)
