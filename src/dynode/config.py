"""
A module meant for the parsing of JSON config files.

All unknown parameters are parsed as they appear after the json.loads() command.
known parameters are identified by their existence in the PARAMETERS list global variable.

For more information read the comment directly above the PARAMETERS list definition.
"""

import datetime
import json
import os
import warnings
from enum import IntEnum
from functools import partial

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as distributions  # type: ignore
import numpyro.distributions.transforms as transforms  # type: ignore
from jax.random import PRNGKey


class Config:
    """
    A Configuration class designed to take JSON config files,
    validate them, and generate downstream parameters where applicable
    """

    def __init__(self, config_json_str) -> None:
        self.add_file(config_json_str)

    def add_file(self, config_json_str):
        """loads a JSON string into self,
        overriding any shared names, asserting valid configuration of parameters,
        and setting any downstream parameters.

        Parameters
        ----------
        config_json_str : str
            JSON string representing a dictionary you wish to add into self

        Returns
        -------
        Config
            self with the parameters from `config_json_str` added on, as well as
            any downstream parameters generated.
        """
        # adds another config to self.__dict__ and resets downstream parameters again
        config = json.loads(
            config_json_str, object_hook=distribution_converter
        )
        config = self.convert_types(config)
        self.__dict__.update(**config)
        self.assert_valid_configuration()
        self.set_downstream_parameters()
        return self

    def asdict(self):
        return self.__dict__

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
            if "downstream" in parameter.keys():
                downstream_function = parameter["downstream"]
                if not isinstance(key, list):
                    key = [key]
                if all([hasattr(self, k) for k in key]):
                    downstream_function(self, key)

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
                try:
                    [
                        (
                            # some validators only take single values not lists
                            val_func(key[0], vals[0])
                            if len(key) == 1
                            else val_func(key, vals)
                        )
                        for val_func in validator_funcs
                    ]
                except Exception as e:
                    if len(key) > 1:
                        err_text = """There was an issue validating your Config object.
                        The error was caused by the intersection of the following parameters: %s.
                        %s""" % (
                            key,
                            e,
                        )
                    else:
                        err_text = """The following error occured while validating the %s
                        parameter in your configuration file: %s""" % (
                            key[0],
                            e,
                        )
                    raise ConfigValidationError(err_text)


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


def set_num_waning_compartments_and_rates(conf, _):
    conf.NUM_WANING_COMPARTMENTS = len(conf.WANING_TIMES)
    # odes often need waning rates not times
    # since last waning compartment set to 0, avoid a div by zero error here
    conf.WANING_RATES = np.array(
        [
            1 / waning_time if waning_time > 0 else 0
            for waning_time in conf.WANING_TIMES
        ]
    )


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


def test_enum_len(key, enum, expected_len):
    assert (
        len(enum) == expected_len
    ), "Expected %s to have %s entries, got %s" % (
        key,
        expected_len,
        len(enum),
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
    """
    compares that vals[0] >= vals[1],
    attempting to compare the upper and lower bounds of
    vals[0] and vals[1] if either or both are distributions.
    some distribution `a` is considered >= distribution `b` if
    a.support.lower_bound >= b.support.upper_bound
    """
    # both keys are distributions, compare the lower vs upper bound
    if issubclass(type(vals[0]), distributions.Distribution) and issubclass(
        type(vals[1]), distributions.Distribution
    ):
        lower_dist = vals[0]
        higher_dist = vals[1]
        if hasattr(higher_dist.support, "lower_bound") and hasattr(
            lower_dist.support, "upper_bound"
        ):
            assert (
                higher_dist.support.lower_bound
                >= lower_dist.support.upper_bound
            ), (
                "the support for the distribution in %s must have a lower bound that is greater than, the upper bound of %s"
                "got %s >= %s. Try specifying a support constraint on the distribution"
                % (
                    keys[0],
                    keys[1],
                    str(higher_dist.support.lower_bound),
                    str(lower_dist.support.ipper_bound),
                )
            )
        elif isinstance(lower_dist.support, distributions.constraints._Real):
            assert False, (
                "the support for the distribution in %s spans the entire number line,"
                "thus you cant guarantee that independent draws from %s will always be greater"
                % (keys[0], keys[1])
            )
        elif isinstance(higher_dist.support, distributions.constraints._Real):
            assert False, (
                "the support for the distribution in %s spans the entire number line,"
                "thus you cant guarantee that independent draws from %s will always be lower"
                % (keys[1], keys[0])
            )
        else:
            warnings.warn(
                "both %s and %s lack support bounds, can not validate the distribution relationship"
                % (keys[0], keys[1])
            )
    # check lower bound is >= some static value
    elif issubclass(type(vals[0]), distributions.Distribution):
        dist = vals[0]
        if hasattr(dist.support, "lower_bound"):
            assert (
                dist.support.lower_bound >= vals[1]
            ), "lower bound of %s must be >= %s, got %s >= %s" % (
                keys[0],
                keys[1],
                str(higher_dist.support.lower_bound),
                str(vals[1]),
            )
        elif isinstance(dist.support, distributions.constraints._Real):
            assert False, (
                "the support for the distribution in %s spans the entire number line,"
                "thus you cant guarantee that independent draws from %s will always be lower"
                % (keys[0], keys[1])
            )
        else:
            warnings.warn(
                "%s lack support bounds, can not validate the distribution relationship"
                % (keys[0])
            )
    # check some static value is always >= upper bound of the distribution
    elif issubclass(type(vals[1]), distributions.Distribution):
        dist = vals[1]
        if hasattr(dist.support, "upper_bound"):
            assert (
                vals[0] >= dist.support.upper_bound
            ), " %s must be >= upper bound of %s, got %s >= %s" % (
                keys[0],
                keys[1],
                str(vals[0]),
                str(dist.support.upper_bound),
            )
        elif isinstance(dist.support, distributions.constraints._Real):
            assert False, (
                "the support for the distribution in %s spans the entire number line,"
                "thus you cant guarantee that %s will always be greater"
                % (keys[1], keys[0])
            )
        else:
            warnings.warn(
                "%s lack support bounds, can not validate the distribution relationship"
                % (keys[1])
            )
    else:  # both static values
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


def test_equal_len(keys, vals):
    test_len(keys, [len(vals[0]), vals[1]])


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
        "name": "VACCINATION_MODEL_DATA",
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
            partial(
                test_type, tested_type=(int, float, distributions.Distribution)
            ),
            test_not_negative,
        ],
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
        "downstream": set_num_waning_compartments_and_rates,
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
        "name": ["NUM_STRAINS", "STRAIN_IDX"],
        # check that len(STRAIN_IDX)==NUM_STRAINS
        "validate": lambda keys, vals: test_enum_len(
            keys[1], vals[1], vals[0]
        ),
    },
    {
        "name": "MAX_VACCINATION_COUNT",
        "validate": test_not_negative,
    },
    {
        "name": "AGE_DOSE_SPECIFIC_VAX_COEF",
        "type": np.array,
        "validate": [
            lambda key, val: test_all_in_list(
                key, val.flatten(), test_not_negative
            ),
        ],
    },
    {  # check that AGE_DOSE_SPECIFIC_VAX_COEF.shape = (NUM_AGE_GROUPS, MAX_VACCINATION_COUNT + 1)
        "name": [
            "AGE_DOSE_SPECIFIC_VAX_COEF",
            "NUM_AGE_GROUPS",
            "MAX_VACCINATION_COUNT",
        ],
        "validate": lambda keys, vals: test_shape(
            keys, ((vals[1], vals[2] + 1), vals[0])
        ),
    },
    {
        "name": "VACCINE_EFF_MATRIX",
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
        "name": "CONSTANT_STEP_SIZE",
        "validate": [
            test_not_negative,
            partial(test_type, tested_type=(int, float)),
        ],
        "type": float,
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
        "name": ["NUM_STRAINS", "MAX_VACCINATION_COUNT", "VACCINE_EFF_MATRIX"],
        # check that VACCINE_EFF_MATRIX shape is (NUM_STRAINS, MAX_VACCINATION_COUNT + 1)
        "validate": lambda key, vals: test_shape(
            key, [(vals[0], vals[1] + 1), vals[2]]
        ),
    },
    {
        "name": "INTRODUCTION_TIMES",
        "validate": [
            partial(test_type, tested_type=list),
            lambda key, val: [
                [test_not_negative(key, intro_time) for intro_time in val]
            ],
        ],
        "downstream": set_num_introduced_strains,
    },
    {
        "name": "INTRODUCTION_SCALES",
        "validate": [
            partial(test_type, tested_type=list),
            lambda key, val: [
                [test_positive(key, intro_scale) for intro_scale in val]
            ],
        ],
    },
    {
        "name": "INTRODUCTION_PCTS",
        "validate": [
            partial(test_type, tested_type=list),
            lambda key, val: [
                [test_not_negative(key, intro_perc) for intro_perc in val]
            ],
        ],
    },
    {
        "name": [
            "INTRODUCTION_TIMES",
            "INTRODUCTION_SCALES",
            "INTRODUCTION_PCTS",
        ],
        "validate": [
            lambda key, val: test_equal_len(
                [key[0], key[1]], [val[0], val[1]]
            ),
            lambda key, val: test_equal_len(
                [key[1], key[2]], [val[1], val[2]]
            ),
            # by transitive property, len(INTRODUCTION_TIMES) == len(INTRODUCTION_PCTS)
        ],
    },
    {
        "name": "SEASONALITY_AMPLITUDE",
        "validate": [
            partial(
                test_type, tested_type=(float, int, distributions.Distribution)
            ),
            # -1.0 <= SEASONALITY_PEAK <= 1.0
            lambda key, val: compare_geq([key, "-1.0"], [val, -1.0]),
            lambda key, val: compare_geq(["1.0", key], [1.0, val]),
        ],
    },
    {
        "name": "SEASONALITY_SECOND_WAVE",
        "validate": [
            partial(
                test_type, tested_type=(float, int, distributions.Distribution)
            ),
            # 0 <= SEASONALITY_SECOND_WAVE <= 1.0
            lambda key, val: compare_geq([key, "0"], [val, 0]),
            lambda key, val: compare_geq(["1.0", key], [1.0, val]),
        ],
    },
    {
        "name": "SEASONALITY_SHIFT",
        "validate": [
            partial(
                test_type, tested_type=(float, int, distributions.Distribution)
            ),
            # -365/2 <= SEASONALITY_SHIFT <= 365/2
            lambda key, val: compare_geq([key, "-365/2"], [val, -182.5]),
            lambda key, val: compare_geq(["365/2", key], [182.5, val]),
        ],
    },
    {
        "name": "INIT_DATE",
        # "validate": do_nothing,
        "type": lambda s: datetime.datetime.strptime(s, "%Y-%m-%d").date(),
    },
    {
        "name": "VACCINATION_SEASON_CHANGE",
        # "validate": do_nothing,
        "type": lambda s: datetime.datetime.strptime(s, "%Y-%m-%d").date(),
    },
    {
        "name": "SEASONAL_VACCINATION",
        "validate": partial(test_type, tested_type=(bool)),
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
        "name": "E_AXIS_IDX",
        # "validate": do_nothing,
        "type": lambda lst: IntEnum("enum", lst, start=0),
    },
    {
        "name": "I_AXIS_IDX",
        # "validate": do_nothing,
        "type": lambda lst: IntEnum("enum", lst, start=0),
    },
    {
        "name": "C_AXIS_IDX",
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
    """A basic class meant to denote when the Config
    class is having an issue parsing a configuration file"""

    pass


class ConfigValidationError(Exception):
    """A basic class meant to denote when the Config
    class is having an issue validating a configuration file"""

    pass