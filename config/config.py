import datetime
import json
import os
import subprocess
from enum import IntEnum

import git
import jax.numpy as jnp


class Config:
    def __init__(self, config_json_path) -> None:
        self.add_file(config_json_path)

    def add_file(self, config_json_path):
        # adds another config to self.__dict__ and resets downstream parameters again
        config = json.load(open(config_json_path, "r"))
        config = self.convert_types(config)
        self.__dict__.update(**config)
        self.assert_valid_configuration()
        self.set_downstream_parameters()
        return self

    def convert_types(self, config):
        """
        takes a dictionary of config parameters, consults the PARAMETERS global list and attempts to convert the type
        of each parameter whos name matches.
        """
        for validator in PARAMETERS:
            key = validator["name"]
            cast_type = validator.get("type", False)
            # if this validator needs to be cast
            if cast_type:
                config_val = config.get(key, False)
                # make sure we actually have the value in our incoming config
                if config_val:
                    config[key] = cast_type(config_val)
        return config

    def set_downstream_parameters(self):
        """
        A parameter that checks if a specific parameter exists, then sets any parameters that depend on it.

        E.g. `NUM_AGE_GROUPS` = len(`AGE_LIMITS`) if `AGE_LIMITS` exists, set `NUM_AGE_GROUPS`
        """
        for validator in PARAMETERS:
            key = validator["name"]
            downstream_function = validator.get("downstream", False)
            # if the key has no downstream functions, dont bother
            if downstream_function:
                # validator requires multiple params checked against eachother
                if isinstance(key, list):
                    if all([hasattr(self, k) for k in key]):
                        downstream_function(self, key)
                # just one param being tested
                else:
                    if hasattr(self, key):
                        downstream_function(self, key)
        self.GIT_HASH = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("ascii")
            .strip()
        )
        self.LOCAL_REPO = git.Repo()

    def assert_valid_configuration(self):
        """
        checks the soundness of parameters passed into Config. Does not check for the existence of certain key parameters
        """
        for validator in PARAMETERS:
            key = validator["name"]
            # validator requires multiple params checked against eachother
            if isinstance(key, list):
                contained_within_config = [hasattr(self, k) for k in key]
                if all(contained_within_config):
                    vals = [getattr(self, k) for k in key]
                    validator["validate"](key, vals)
            # just one param being tested
            else:
                if hasattr(self, key):
                    validator["validate"](key, getattr(self, key))


def set_downstream_age_variables(conf, _):
    """
    given AGE_LIMITS, set downstream variables from there
    """
    conf.NUM_AGE_GROUPS = len(conf.AGE_LIMITS)

    conf.AGE_GROUP_STRS = [
        str(conf.AGE_LIMITS[i - 1]) + "-" + str(conf.AGE_LIMITS[i] - 1)
        for i in range(1, len(conf.AGE_LIMITS))
    ] + [str(conf.AGE_LIMITS[-1]) + "+"]

    conf.AGE_GROUP_IDX = IntEnum("age", conf.AGE_GROUP_STRS, start=0)


def set_num_introduced_strains(conf, _):
    """
    given INTRODUCTION_TIMES, set downstream variables from there
    """
    conf.NUM_INTRODUCED_STRAINS = len(conf.INTRODUCTION_TIMES)


def set_wane_enum(conf, _):
    """
    given NUM_WANING_COMPARTMENTS set the WANE_IDX
    """
    conf.WANE_IDX = IntEnum(
        "w_idx",
        ["W" + str(idx) for idx in range(conf.NUM_WANING_COMPARTMENTS)],
        start=0,
    )


def path_checker(key, value):
    assert os.path.exists(value), "%s : %s is not a valid path" % (key, value)


def test_positive(key, value):
    assert value > 0, "%s must be greater than zero, got %s" % (
        key,
        str(value),
    )


def test_not_negative(key, value):
    assert value >= 0, "%s must be non-negative, got %s" % (
        key,
        str(value),
    )


def age_limit_checks(key, age_limits):
    assert all(
        [
            age_limits[idx] > age_limits[idx - 1]
            for idx in range(1, len(age_limits))
        ]
    ), ("%s must be strictly increasing" % key)
    assert (
        age_limits[-1] < 85
    ), "age limits can not exceed 84 years of age, the last age bin is implied and does not need to be included"


def compare_geq(keys, vals):
    key1, key2 = keys[0], keys[1]
    val1, val2 = vals[0], vals[1]
    assert val1 >= val2, "%s must be >= %s, however got %d >= %d" % (
        key1,
        key2,
        val1,
        val2,
    )


def test_type(key, val, tested_type):
    assert isinstance(val, tested_type), "%s must be an %s, found %s"(
        key, str(tested_type), str(type(val))
    )


def test_non_empty(key, val):
    assert len(val) > 0, "%s is expected to be a non-empty list" % key


def test_len(keys, vals):
    key1, key2 = keys[0], keys[1]
    len_of_array, array = vals[0], vals[1]
    assert len_of_array == len(array), "len(%s) must equal to %s" % (
        key2,
        key1,
    )


def test_shape(keys, vals):
    key1, key2 = keys[0], keys[1]
    shape_of_matrix, array = vals[0], vals[1]
    assert shape_of_matrix == array.shape, "%s.shape must equal to %s" % (
        key2,
        key1,
    )


def test_ascending(key, lst):
    assert all(
        [wane_time >= 1 for wane_time in lst[:-1]]
    ), "Can not have waning time less than 1 day, time is in days if you meant to put months"


def test_zero(key, val):
    assert val == 0, "value in %s must be zero" % key


def do_nothing(key, val):
    return key, val


PARAMETERS = [
    {
        "name": "SAVE_PATH",
        "validate": path_checker,
    },
    {
        "name": "DEMOGRAPHIC_DATA_PATH",
        "validate": path_checker,
    },
    {
        "name": "SEROLOGICAL_DATA_PATH",
        "validate": path_checker,
    },
    {
        "name": "SIM_DATA_PATH",
        "validate": path_checker,
    },
    {
        "name": "VAX_MODEL_DATA",
        "validate": path_checker,
    },
    {
        "name": "AGE_LIMITS",
        "validate": age_limit_checks,
        "downstream": set_downstream_age_variables,
    },
    {
        "name": "POP_SIZE",
        "validate": test_positive,
    },
    {
        "name": "INITIAL_INFECTIONS",
        "validate": test_not_negative,
    },
    {
        "name": ["POP_SIZE", "INITIAL_INFECTIONS"],
        "validate": compare_geq,
    },
    {
        "name": "INFECTIOUS_PERIOD",
        "validate": test_not_negative,
    },
    {
        "name": "EXPOSED_TO_INFECTIOUS",
        "validate": test_not_negative,
    },
    {
        "name": "STRAIN_SPECIFIC_R0",
        "validate": test_non_empty,
        "type": jnp.array,
    },
    {
        "name": "NUM_WANING_COMPARTMENTS",
        "validate": test_positive,
        "downstream": set_wane_enum,
    },
    {
        "name": "WANING_TIMES",
        "validate": lambda key, vals: [
            test_positive(key, val) for val in vals[:-1]
        ],
    },
    {
        "name": "WANING_TIMES",
        "validate": lambda key, vals: test_zero(key, vals[-1]),
    },
    {
        "name": "WANING_TIMES",
        "validate": lambda key, vals: [
            test_type(key, val, int) for val in vals
        ],
    },
    {
        "name": "WANING_PROTECTIONS",
        "validate": lambda key, vals: [
            test_not_negative(key, val) for val in vals
        ],
        "type": jnp.array,
    },
    {
        "name": ["NUM_WANING_COMPARTMENTS", "WANING_TIMES"],
        "validate": test_len,
    },
    {
        "name": ["NUM_WANING_COMPARTMENTS", "WANING_PROTECTIONS"],
        "validate": test_len,
    },
    {
        "name": "STRAIN_INTERACTIONS",
        "validate": test_non_empty,
        "type": jnp.array,
    },
    {
        "name": ["NUM_STRAINS", "STRAIN_INTERACTIONS"],
        # check that STRAIN_INTERACTIONS shape is (NUM_STRAINS, NUM_STRAINS)
        "validate": lambda key, vals: test_shape(
            key, [(vals[0], vals[0]), vals[1]]
        ),
    },
    {
        "name": ["NUM_STRAINS", "CROSSIMMUNITY_MATRIX"],
        # check that CROSSIMMUNITY_MATRIX shape is (NUM_STRAINS, 2**NUM_STRAINS)
        "validate": lambda key, vals: test_shape(
            key, [(vals[0], 2 ** vals[0]), vals[1]]
        ),
    },
    {
        "name": "MAX_VAX_COUNT",
        "validate": test_not_negative,
    },
    {
        "name": "VAX_EFF_MATRIX",
        "validate": test_non_empty,
        "type": jnp.array,
    },
    {
        "name": "BETA_TIMES",
        "validate": lambda key, lst: [
            test_not_negative(key, beta_time) for beta_time in lst
        ],
        "type": jnp.array,
    },
    {
        "name": "BETA_COEFICIENTS",
        "validate": lambda key, lst: [
            test_not_negative(key, beta_time) for beta_time in lst
        ],
        "type": jnp.array,
    },
    {
        "name": "STRAIN_R0s",
        "validate": lambda key, lst: [
            test_not_negative(key, r0) for r0 in lst
        ],
        "type": jnp.array,
    },
    {
        "name": ["NUM_STRAINS", "MAX_VAX_COUNT", "VAX_EFF_MATRIX"],
        # check that VAX_EFF_MATRIX shape is (NUM_STRAINS, MAX_VAX_COUNT + 1)
        "validate": lambda key, vals: test_shape(
            key, [(vals[0], vals[1] + 1), vals[2]]
        ),
    },
    {
        "name": "INTRODUCTION_TIMES",
        "validate": lambda key, val: [
            [test_not_negative(key, intro_time) for intro_time in val]
        ],
        "downstream": set_num_introduced_strains,
    },
    {
        "name": "INIT_DATE",
        "validate": do_nothing,
        "type": lambda s: datetime.datetime.strptime(s, "%Y-%m-%d"),
    },
    {
        "name": "COMPARTMENT_IDX",
        "validate": do_nothing,
        "type": lambda lst: IntEnum("enum", lst, start=0),
    },
    {
        "name": "S_AXIS_IDX",
        "validate": do_nothing,
        "type": lambda lst: IntEnum("enum", lst, start=0),
    },
    {
        "name": "I_AXIS_IDX",
        "validate": do_nothing,
        "type": lambda lst: IntEnum("enum", lst, start=0),
    },
    {
        "name": "STRAIN_IDX",
        "validate": do_nothing,
        "type": lambda lst: IntEnum("enum", lst, start=0),
    },
]
