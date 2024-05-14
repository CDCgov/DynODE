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


def test_invalid_seasonality_amplitude_type():
    input_json = """{"SEASONALITY_AMPLITUDE": [0]}"""
    with pytest.raises(AssertionError):
        Config(input_json)


def test_invalid_seasonality_amplitude_val():
    input_json = """{"SEASONALITY_AMPLITUDE": 4.0}"""
    with pytest.raises(AssertionError):
        Config(input_json)


def test_invalid_vax_age_coefs_type():
    input_json = """{"AGE_DOSE_SPECIFIC_VAX_COEF": "blah"}"""
    with pytest.raises(AssertionError):
        Config(input_json)


def test_invalid_vax_path():
    input_json = """{"VAX_MODEL_DATA": "blah"}"""
    with pytest.raises(AssertionError):
        Config(input_json)


def test_invalid_vax_age_coefs_vals():
    # will fail because we have 4 rows instead of 3
    input_json = """{"AGE_DOSE_SPECIFIC_VAX_COEF":
                        [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
                    "NUM_AGE_GROUPS":3,
                    "MAX_VAX_COUNT": 2
                        }"""
    with pytest.raises(AssertionError):
        Config(input_json)


def test_valid_vax_age_coefs():
    input_json = """{"AGE_DOSE_SPECIFIC_VAX_COEF":
                        [[1, 1, 1], [1, 1, 1]],
                    "NUM_AGE_GROUPS":2,
                    "MAX_VAX_COUNT": 2
                        }"""
    assert Config(input_json).AGE_DOSE_SPECIFIC_VAX_COEF.tolist() == [
        [1, 1, 1],
        [1, 1, 1],
    ]


def test_invalid_seasonality_amplitude_val_negative():
    input_json = """{"SEASONALITY_AMPLITUDE": -4.0}"""
    with pytest.raises(AssertionError):
        Config(input_json)


def test_invalid_seasonality_amplitude_dist():
    input_json = """{"SEASONALITY_AMPLITUDE": {
        "distribution": "Normal",
        "params": {
            "scale": 1,
            "loc": 0
        }
    }}"""
    with pytest.raises(AssertionError):
        Config(input_json)


def test_valid_seasonality_amplitude():
    input_json = """{"SEASONALITY_AMPLITUDE": 0.15}"""
    assert Config(input_json).SEASONALITY_AMPLITUDE == 0.15


def test_valid_seasonality_amplitude_dist():
    input_json = """{"SEASONALITY_AMPLITUDE": {
        "distribution": "Beta",
        "params": {
            "concentration1": 1,
            "concentration0": 19
        }
    }}"""
    assert isinstance(Config(input_json).SEASONALITY_AMPLITUDE, dist.Beta)


def test_invalid_seasonality_second_wave_type():
    input_json = """{"SEASONALITY_SECOND_WAVE": [0]}"""
    with pytest.raises(AssertionError):
        Config(input_json)


def test_invalid_seasonality_second_wave_val():
    input_json = """{"SEASONALITY_SECOND_WAVE": 1.5}"""
    with pytest.raises(AssertionError):
        Config(input_json)


def test_invalid_seasonality_second_wave_val_negative():
    input_json = """{"SEASONALITY_SECOND_WAVE": -1.5}"""
    with pytest.raises(AssertionError):
        Config(input_json)


def test_invalid_seasonalit_second_wave_dist():
    input_json = """{"SEASONALITY_SECOND_WAVE": {
        "distribution": "Normal",
        "params": {
            "scale": 1,
            "loc": 0
        }
    }}"""
    with pytest.raises(AssertionError):
        Config(input_json)


def test_valid_seasonality_second_wave():
    input_json = """{"SEASONALITY_SECOND_WAVE": 0.15}"""
    assert Config(input_json).SEASONALITY_SECOND_WAVE == 0.15


def test_valid_seasonality_second_wave_dist():
    input_json = """{"SEASONALITY_SECOND_WAVE":{
        "distribution": "Beta",
        "params": {
            "concentration1": 1,
            "concentration0": 19
        }
    }}"""
    assert isinstance(Config(input_json).SEASONALITY_SECOND_WAVE, dist.Beta)


def test_invalid_seasonality_shift_type():
    input_json = """{"SEASONALITY_SHIFT": [0]}"""
    with pytest.raises(AssertionError):
        Config(input_json)


def test_invalid_seasonality_shift_val():
    input_json = """{"SEASONALITY_SHIFT": 183}"""
    with pytest.raises(AssertionError):
        Config(input_json)


def test_invalid_seasonality_shift_val_negative():
    input_json = """{"SEASONALITY_SHIFT": -183}"""
    with pytest.raises(AssertionError):
        Config(input_json)


def test_invalid_seasonalit_shift_dist():
    input_json = """{"SEASONALITY_SHIFT": {
        "distribution": "Normal",
        "params": {
            "scale": 1,
            "loc": 0
        }
    }}"""
    with pytest.raises(AssertionError):
        Config(input_json)


def test_valid_seasonality_shift():
    input_json = """{"SEASONALITY_SHIFT": 10}"""
    assert Config(input_json).SEASONALITY_SHIFT == 10


def test_valid_seasonality_shift_dist():
    input_json = """{"SEASONALITY_SHIFT":{
        "distribution": "TruncatedNormal",
        "params": {
            "loc": 50,
            "scale": 20,
            "low": -180,
            "high": 180
        }
    }}"""
    assert issubclass(
        dist.TwoSidedTruncatedDistribution,
        type(Config(input_json).SEASONALITY_SHIFT),
    )


def test_invalid_introduction_perc_type():
    input_json = """{"INTRODUCTION_PERCS": 0.1}"""
    with pytest.raises(AssertionError):
        Config(input_json)


def test_invalid_introduction_perc_val():
    input_json = """{"INTRODUCTION_PERCS":[-1]}"""
    with pytest.raises(AssertionError):
        Config(input_json)


def test_valid_introduction_perc():
    input_json = """{"INTRODUCTION_PERCS": [0.1]}"""
    assert Config(input_json).INTRODUCTION_PERCS == [0.1]


def test_invalid_introduction_times_type():
    input_json = """{"INTRODUCTION_TIMES": 0}"""
    with pytest.raises(AssertionError):
        Config(input_json)


def test_invalid_introduction_times_val():
    input_json = """{"INTRODUCTION_TIMES": [-1]}"""
    with pytest.raises(AssertionError):
        Config(input_json)


def test_valid_introduction_times_val():
    input_json = """{"INTRODUCTION_TIMES": [10]}"""
    assert Config(input_json).INTRODUCTION_TIMES == [10]


def test_invalid_introduction_scale_type():
    input_json = """{"INTRODUCTION_SCALES": 5}"""
    with pytest.raises(AssertionError):
        Config(input_json)


def test_invalid_introduction_scale_val():
    input_json = """{"INTRODUCTION_SCALES": [-1]}"""
    with pytest.raises(AssertionError):
        Config(input_json)


def test_valid_introduction_scale_val():
    input_json = """{"INTRODUCTION_SCALES": [10]}"""
    assert Config(input_json).INTRODUCTION_SCALES == [10]


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


def test_negative_initial_infections_scale():
    input_json = """{"INITIAL_INFECTIONS_SCALE": -1.2}"""
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


def test_invalid_step_size():
    input_json = """{"CONSTANT_STEP_SIZE": -1.0}"""
    with pytest.raises(AssertionError):
        Config(input_json)


def test_valid_step_size():
    input_json = """{"CONSTANT_STEP_SIZE": 0.0}"""
    c = Config(input_json)
    assert c.CONSTANT_STEP_SIZE == 0.0


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
