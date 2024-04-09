import datetime
import json
import os
import warnings
from enum import IntEnum
from functools import partial

import git
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as distributions
import numpyro.distributions.transforms as transforms
from jax.random import PRNGKey


class Config:
    def __init__(self, config_json_str) -> None:
        self.add_file(config_json_str)

    def add_file(self, config_json_str):
        # adds another config to self.__dict__ and resets downstream parameters again
        config = json.loads(
            config_json_str, object_hook=distribution_converter
        )
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
        for parameter in PARAMETERS:
            key = parameter["name"]
            # if this validator needs to be cast
            if "type" in parameter.keys():
                cast_type = parameter["type"]
                # make sure we actually have the value in our incoming config
                if key in config.keys():
                    try:
                        config[key] = cast_type(config[key])
                    except Exception as e:
                        raise ConfigParserError(
                            "Unable to cast %s into %s" % (key, str(cast_type))
                        ) from e
        return config

    def set_downstream_parameters(self):
        """
        A function that checks if a specific parameter exists, then sets any parameters that depend on it.

        E.g., `NUM_AGE_GROUPS` = len(`AGE_LIMITS`) if `AGE_LIMITS` exists, set `NUM_AGE_GROUPS`
        """
        for parameter in PARAMETERS:
            key = parameter["name"]
            # if the key has no downstream functions, do nothing
            if "downstream" in parameter.keys():
                downstream_function = parameter["downstream"]
                # turn into list of len(1) if not already
                if not isinstance(key, list):
                    key = [key]
                # dont try to create downstream unless config has all necessary keys
                if all([hasattr(self, k) for k in key]):
                    downstream_function(self, key)
        # take note of the current git hash for reproducibility reasons
        self.LOCAL_REPO = git.Repo()
        self.GIT_HASH = self.LOCAL_REPO.head.object.hexsha

    def assert_valid_configuration(self):
        """
        checks the soundness of parameters passed into Config by referencing the name of parameters passed to the config
        with the PARAMETERS global variable. If a distribution is passed instead of a value, blindly accepts the distribution.

        Raises assert errors if parameter(s) are incongruent in some way.
        """
        for param in PARAMETERS:
            key = param["name"]
            key = make_list_if_not(key)
            validator_funcs = param.get("validate", False)
            # if there are validators to test, and the key(s) are found in our config, lets test them
            if validator_funcs and all([hasattr(self, k) for k in key]):
                validator_funcs = make_list_if_not(validator_funcs)
                vals = [getattr(self, k) for k in key]
                # val_func() throws assert errors if incongruence arrises
                [
                    (
                        val_func(key[0], vals[0])
                        if len(key) == 1  # convert back to floats if needed
                        else val_func(key, vals)
                    )
                    for val_func in validator_funcs
                ]


def make_list_if_not(obj):
    return obj if isinstance(obj, (list, np.ndarray)) else [obj]


def distribution_converter(dct):
    """
    Converts a distribution or transform as specified in JSON config file into
    a numpyro distribution/transform object.
    This function is called as a part of json.loads(object_hook=distribution_converter)
    meaning it executes on EVERY JSON object within a JSON string,
    recursively from innermost nested outwards.


    a distribution is identified by the `distribution` and `params` keys inside of a json object
    while a transform is identified by the `transform` and `params` keys inside of a json object
    and a constraint is identified by the `constraint` and `params` keys inside of a json object

    PARAMETERS
    ----------
    `dct`: dict
        A dictionary representing any JSON object that is passed into `Config`.
        Including nested JSON objects which are executed from deepest nested outwards.

    Returns
    -----------
    dict or numpyro.distributions object. If `distribution_converter` identifies that dct is a valid JSON representation of a
    numpyro distribution or transform, it will return it. Otherwise it returns dct unmodified.
    """
    try:
        if "distribution" in dct.keys() and "params" in dct.keys():
            numpyro_dst = dct["distribution"]
            numpyro_dst_params = dct["params"]
            if numpyro_dst in distribution_types.keys():
                distribution = distribution_types[numpyro_dst](
                    **numpyro_dst_params
                )
                # numpyro does lazy eval of distributions, if the user passes in invalid parameter values
                # they wont be caught until runtime, so we sample here to raise an error
                _ = distribution.sample(PRNGKey(1))
                return distribution
            else:
                raise KeyError(
                    "The distribution name was not found in the available distributions, "
                    "see distribution names here: https://num.pyro.ai/en/stable/distributions.html#distributions"
                )
        elif "transform" in dct.keys() and "params" in dct.keys():
            numpyro_transform = dct["transform"]
            numpyro_transform_params = dct["params"]
            if numpyro_transform in transform_types.keys():
                transform = transform_types[numpyro_transform](
                    **numpyro_transform_params
                )
                return transform
            else:
                raise KeyError(
                    "The transform name was not found in the available transformations, "
                    "see transform names here: https://num.pyro.ai/en/stable/distributions.html#transforms"
                )
        elif "constraint" in dct.keys():
            numpyro_constraint = dct["constraint"]
            if numpyro_constraint in constraint_types.keys():
                # some constraints are not callable, like unit_interval
                if (
                    "params" not in dct.keys()
                    or len(dct["params"].keys()) == 0
                ):
                    constraint = constraint_types[numpyro_constraint]
                else:
                    numpyro_constraint_params = dct["params"]
                    constraint = constraint_types[numpyro_constraint](
                        **numpyro_constraint_params
                    )
                return constraint
            else:
                raise KeyError(
                    "The constraint name was not found in the available constraints, "
                    "see constraint names here: https://num.pyro.ai/en/stable/_modules/numpyro/distributions/constraints.html"
                )
    except Exception as e:
        # reraise the error
        raise ConfigParserError(
            "There was an error parsing the following distribution/transformation: %s \n "
            "see docs to make sure you didnt misspell something: https://num.pyro.ai/en/stable/distributions.html#distributions \n"
            "or you may have passed incorrect parameters types/names into the distribution"
            % str(dct)
        ) from e
    # do nothing if this isnt a distribution or transform
    return dct


#############################################################################
#######################DOWNSTREAM/VALIDATION FUNCTIONS#######################
#############################################################################
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


def set_num_waning_compartments(conf, _):
    conf.NUM_WANING_COMPARTMENTS = len(conf.WANING_TIMES)


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
    """
    checks if a value is positive.
    If `value` is a distribution, checks that the lower bound of its support is positive
    """
    if issubclass(type(value), distributions.Distribution):
        if hasattr(value.support, "lower_bound"):
            assert value.support.lower_bound > 0, (
                "the support for the distribution in %s must have a lower bound greater than zero, "
                "got %s. Try specifying a support constraint on the distribution"
                % (
                    key,
                    str(value.support.lower_bound),
                )
            )
        elif isinstance(value.support, distributions.constraints._Real):
            assert False, (
                "the support for the distribution in %s must have a lower bound greater than zero,"
                "got all Real numbers. Try specifying a support constraint on the distribution"
                % key
            )
        else:
            warnings.warn(
                "%s does not have a support lower bound, can not validate the distribution"
                % key
            )
    else:  # not a distribution, just a value
        assert value > 0, "%s must be greater than zero, got %s" % (
            key,
            str(value),
        )


def test_not_negative(key, value):
    """
    checks if a value is not negative.
    If `value` is a distribution, checks that the lower bound of its support not negative
    """
    if issubclass(type(value), distributions.Distribution):
        if hasattr(value.support, "lower_bound"):
            assert value.support.lower_bound >= 0, (
                "the support for the distribution in %s must have a lower bound that is non-negative, "
                "got %s. Try specifying a support constraint on the distribution"
                % (
                    key,
                    str(value.support.lower_bound),
                )
            )
        elif isinstance(value.support, distributions.constraints._Real):
            assert False, (
                "the support for the distribution in %s must have a lower bound greater than zero,"
                "got all Real numbers. Try specifying a support constraint on the distribution"
                % key
            )
        else:
            warnings.warn(
                "%s does not have a support lower bound, can not validate the distribution"
                % key
            )
    else:  # not a distribution, just a value
        assert value >= 0, "%s must be non-negative, got %s" % (
            key,
            str(value),
        )


def test_all_in_list(key, lst, func):
    """
    a function which tests a different constraint function defined in this file across all values of a list
    """
    try:
        for i, value in enumerate(lst):
            func(key, value)
    except Exception as e:
        # reraise exception specifying which index failed the test
        raise AssertionError(
            "The %s'th element of the %s array failed the above test"
            % (str(i), key)
        ) from e


def age_limit_checks(key, age_limits):
    test_not_negative(key, age_limits[0])
    test_ascending(key, age_limits)
    assert all(
        [isinstance(a, int) for a in age_limits]
    ), "ages must be int, not float because census age data is specified as int"
    assert age_limits[-1] < MAX_AGE_CENSUS_DATA, (
        "age limits can not exceed "
        + str(MAX_AGE_CENSUS_DATA)
        + " years of age, the last age bin is implied and does not need to be included"
    )


def compare_geq(keys, vals):
    assert vals[0] >= vals[1], "%s must be >= %s, however got %d >= %d" % (
        keys[0],
        keys[1],
        vals[0],
        vals[1],
    )


def test_type(key, val, tested_type):
    assert isinstance(val, tested_type) or issubclass(
        type(val), tested_type
    ), "%s must be an %s, found %s" % (
        key,
        str(tested_type),
        str(type(val)),
    )


def test_non_empty(key, val):
    assert len(val) > 0, "%s is expected to be a non-empty list" % key


def test_len(keys, vals):
    assert vals[0] == len(vals[1]), "len(%s) must equal to %s" % (
        keys[1],
        keys[0],
    )


def test_shape(keys, vals):
    key1, key2 = keys[0], keys[1]
    shape_of_matrix, array = vals[0], vals[1]
    assert shape_of_matrix == array.shape, "%s.shape must equal to %s" % (
        key2,
        key1,
    )


def test_ascending(key, lst):
    assert all([lst[idx - 1] < lst[idx] for idx in range(1, len(lst))]), (
        "%s must be placed in increasing order" % key
    )


def test_zero(key, val):
    assert val == 0, "value in %s must be zero" % key


# look at numpyro.distributions, copy over all the distribution names into the dictionary, along with their class constructors.
distribution_types = {
    dist_name: distributions.__dict__.get(dist_name)
    for dist_name in distributions.__all__
}
transform_types = {
    transform_name: transforms.__dict__.get(transform_name)
    for transform_name in transforms.__all__
}
constraint_types = {
    constraint_name: distributions.constraints.__dict__.get(constraint_name)
    for constraint_name in distributions.constraints.__all__
}


#############################################################################
###############################PARAMETERS####################################
#############################################################################
"""
PARAMETERS:
A list of possible parameters contained within any Config file that a model may be expected to read in.
name: the parameter name as written in the JSON config or a list of parameter names.
      if isinstance(name, list) all parameter names must be present before any other sections are executed.
validate: a single function, or list of functions, each with a signature of f(str, obj) -> None
          that raise assertion errors if their conditions are not met.
          Note: ALL validators must pass for Config to accept the parameter
          For the case of test_type, the type of the parameter may be ANY of the tested_type dtypes.
type: If the parameter type is a non-json primative type, specify a function that takes in the nearest JSON primative type and does
      the type conversion. E.G: np.array recieves a JSON primative (list) and returns a numpy array.
downstream: if receiving this parameter kicks off downstream parameters to be modified or created, a function which takes the Config()
            class is accepted to modify/create the downstream parameters.

Note about partial(): the partial function creates an anonymous function, taking a named function as input as well as some
key word arguments. This allows us to pre-specify certain arguments, and allow the parser to pass in the needed ones at runtime.
"""
MAX_AGE_CENSUS_DATA = 85
PARAMETERS = [
    {
        "name": "SAVE_PATH",
        "validate": [partial(test_type, tested_type=str), path_checker],
    },
    {
        "name": "DEMOGRAPHIC_DATA_PATH",
        "validate": [partial(test_type, tested_type=str), path_checker],
    },
    {
        "name": "SEROLOGICAL_DATA_PATH",
        "validate": [partial(test_type, tested_type=str), path_checker],
    },
    {
        "name": "SIM_DATA_PATH",
        "validate": [partial(test_type, tested_type=str), path_checker],
    },
    {
        "name": "VAX_MODEL_DATA",
        "validate": [partial(test_type, tested_type=str), path_checker],
    },
    {
        "name": "AGE_LIMITS",
        "validate": [partial(test_type, tested_type=list), age_limit_checks],
        "downstream": set_downstream_age_variables,
    },
    {
        "name": "POP_SIZE",
        "validate": [partial(test_type, tested_type=int), test_positive],
    },
    {
        "name": "INITIAL_INFECTIONS",
        "validate": [
            partial(test_type, tested_type=(int, float)),
            test_not_negative,
        ],
    },
    {
        "name": "INITIAL_INFECTIONS_SCALE",
        "validate": [
            partial(test_type, tested_type=(int, float)),
            test_not_negative,
        ],
        "type": float,
    },
    {
        "name": ["POP_SIZE", "INITIAL_INFECTIONS"],
        "validate": compare_geq,
    },
    {
        "name": "INFECTIOUS_PERIOD",
        "validate": [
            partial(
                test_type, tested_type=(int, float, distributions.Distribution)
            ),
            test_not_negative,
        ],
    },
    {
        "name": "EXPOSED_TO_INFECTIOUS",
        "validate": [
            partial(
                test_type, tested_type=(int, float, distributions.Distribution)
            ),
            test_not_negative,
        ],
    },
    {
        "name": "WANING_TIMES",
        "validate": [
            partial(test_type, tested_type=list),
            lambda key, vals: [test_positive(key, val) for val in vals[:-1]],
            lambda key, vals: test_zero(key, vals[-1]),
            lambda key, vals: [test_type(key, val, int) for val in vals],
        ],
        "downstream": set_num_waning_compartments,
    },
    {
        "name": "NUM_WANING_COMPARTMENTS",
        "validate": [
            partial(test_type, tested_type=int),
            test_positive,
        ],
        "downstream": set_wane_enum,
    },
    {
        "name": "WANING_PROTECTIONS",
        "validate": lambda key, vals: [
            test_not_negative(key, val) for val in vals
        ],
        "type": np.array,
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
        "type": np.array,
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
        "type": np.array,
    },
    {
        "name": "BETA_TIMES",
        "validate": lambda key, lst: [
            test_not_negative(key, beta_time) for beta_time in lst
        ],
        "type": np.array,
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
        "validate": [
            partial(test_type, tested_type=np.ndarray),
            test_non_empty,
            partial(test_all_in_list, func=test_not_negative),
        ],
        "type": np.array,
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
        # "validate": do_nothing,
        "type": lambda s: datetime.datetime.strptime(s, "%Y-%m-%d").date(),
    },
    {
        "name": "COMPARTMENT_IDX",
        # "validate": do_nothing,
        "type": lambda lst: IntEnum("enum", lst, start=0),
    },
    {
        "name": "S_AXIS_IDX",
        # "validate": do_nothing,
        "type": lambda lst: IntEnum("enum", lst, start=0),
    },
    {
        "name": "I_AXIS_IDX",
        # "validate": do_nothing,
        "type": lambda lst: IntEnum("enum", lst, start=0),
    },
    {
        "name": "STRAIN_IDX",
        # "validate": do_nothing,
        "type": lambda lst: IntEnum("enum", lst, start=0),
    },
    {
        "name": "MAX_TREE_DEPTH",
        "validate": [
            partial(test_type, tested_type=(int)),
            test_positive,
        ],
    },
]


class ConfigParserError(Exception):
    pass
