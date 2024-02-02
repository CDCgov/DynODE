"""
this class acts as a wrapper around the JSON config files, 
casting to appropriate types as well as creating additional variables 
based on the values of provided ones.
"""
import os
import json
import numpy as np
import jax.numpy as jnp
from enum import IntEnum
from numpyro import distributions
import datetime
from functools import partial


class ConfigParser:
    def __init__(self, config_path):
        assert os.path.exists(config_path), (
            "%s is not a valid path" % config_path
        )
        assert ".json" == config_path[-5:], "%s is not in JSON file format"
        self.TYPE_DICT = {
            "date": lambda s: datetime.datetime.strptime(s, "%Y-%m-%d"),
            "enum": lambda lst: IntEnum("enum", lst, start=0),
            "distributions.Normal": lambda mean, sd: distributions.Normal(
                mean, sd
            ),
            "numpy": lambda lst: jnp.array(lst),
        }
        # all JSON objects are passed to self.cast_type
        self.config = json.load(
            open(config_path, "r"), object_hook=self.cast_type
        )

    def get_config(self):
        return self.config

    def cast_type(self, val):
        # return_dict = {}
        if isinstance(val, dict):
            # check if this is a complex type object, return parsed version
            if "__type__" in val.keys() and "__val__" in val.keys():
                return self.complex_type(val["__type__"], val["__val__"])
            # else json object_hook recurses for us
        # # base case primitative types, list, str, int, float
        if isinstance(val, list):
            return jnp.array(val)
        else:  # if not dict, its one of the other primitive types
            return val

    def complex_type(self, type, val):
        """
        parses complex type objects and returns the correct python objects according to the mapping described in TYPE_DICT.
        raises a type error if a type is passed that is not covered in TYPE_DICT.
        """
        # TODO add complex distributions type casting
        if type in self.TYPE_DICT:
            return self.TYPE_DICT[type](val)
        else:
            raise TypeError("Type passed in config not able to be parsed")
