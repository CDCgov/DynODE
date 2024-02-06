from config.config_parser import ConfigParser
from enum import IntEnum
import numpy as np
import os
import subprocess
import git


class Config:
    def __init__(self, config_json_path) -> None:
        config_file = ConfigParser(config_json_path)
        self.__dict__.update(**config_file.get_config())
        self.set_downstream_parameters()

    def add_file(self, config_json_path):
        # adds another config to self.__dict__ and resets downstream parameters again
        config_file = ConfigParser(config_json_path)
        self.__dict__.update(**config_file.get_config())
        self.set_downstream_parameters()
        return self

    def set_downstream_parameters(self):
        """
        A parameter that checks if a specific parameter exists, then sets any parameters that depend on it.

        E.g. `NUM_AGE_GROUPS` = len(`AGE_LIMITS`) if `AGE_LIMITS` exists, set `NUM_AGE_GROUPS`
        """
        if hasattr(self, "AGE_LIMITS"):
            self.NUM_AGE_GROUPS = len(self.AGE_LIMITS)

            self.AGE_GROUP_STRS = [
                str(self.AGE_LIMITS[i - 1]) + "-" + str(self.AGE_LIMITS[i] - 1)
                for i in range(1, len(self.AGE_LIMITS))
            ] + [str(self.AGE_LIMITS[-1]) + "+"]

            self.AGE_GROUP_IDX = IntEnum("age", self.AGE_GROUP_STRS, start=0)

        if hasattr(self, "NUM_WANING_COMPARTMENTS"):
            self.W_IDX = IntEnum(
                "w_idx",
                [
                    "W" + str(idx)
                    for idx in range(self.NUM_WANING_COMPARTMENTS)
                ],
                start=0,
            )
        if hasattr(self, "INTRODUCTION_TIMES"):
            self.NUM_INTRODUCED_STRAINS = len(self.INTRODUCTION_TIMES)
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
        if hasattr(self, "SAVE_PATH"):
            assert os.path.exists(self.SAVE_PATH), (
                "%s is not a valid path" % self.SAVE_PATH
            )
        if hasattr(self, "DEMOGRAPHIC_DATA_PATH"):
            assert os.path.exists(self.DEMOGRAPHIC_DATA_PATH), (
                "%s is not a valid path" % self.DEMOGRAPHIC_DATA
            )
        if hasattr(self, "SEROLOGICAL_DATA_PATH"):
            assert os.path.exists(self.SEROLOGICAL_DATA_PATH), (
                "%s is not a valid path" % self.SEROLOGICAL_DATA
            )
        if hasattr(self, "SIM_DATA_PATH"):
            assert os.path.exists(self.SIM_DATA_PATH), (
                "%s is not a valid path" % self.SIM_DATA
            )
        if hasattr(self, "VAX_MODEL_DATA"):
            assert os.path.exists(self.VAX_MODEL_DATA), (
                "%s is not a valid path" % self.VAX_MODEL_DATA
            )
        if hasattr(self, "AGE_LIMITS"):
            assert all(
                [
                    self.AGE_LIMITS[idx] > self.AGE_LIMITS[idx - 1]
                    for idx in range(1, len(self.AGE_LIMITS))
                ]
            ), "AGE_LIMITS must be strictly increasing"
            assert (
                self.AGE_LIMITS[-1] < 85
            ), "age limits can not exceed 84 years of age, the last age bin is implied and does not need to be included"
        if hasattr(self, "POP_SIZE"):
            assert (
                self.POP_SIZE > 0
            ), "population size must be a non-zero value"
        if hasattr(self, "INITIAL_INFECTIONS"):
            if self.INITIAL_INFECTIONS:
                assert (
                    self.INITIAL_INFECTIONS <= self.POP_SIZE
                ), "cant have more initial infections than total population size"

                assert (
                    self.INITIAL_INFECTIONS >= 0
                ), "cant have negative initial infections"

        if hasattr(self, "INITIAL_POPULATION_FRACTIONS"):
            if self.INITIAL_POPULATION_FRACTIONS:
                assert self.INITIAL_POPULATION_FRACTIONS.shape == (
                    self.NUM_AGE_GROUPS,
                ), (
                    "INITIAL_POPULATION_FRACTIONS must be of shape %s, received %s"
                    % (
                        str((self.NUM_AGE_GROUPS,)),
                        str(self.INITIAL_POPULATION_FRACTIONS.shape),
                    )
                )
                assert (
                    sum(self.INITIAL_POPULATION_FRACTIONS) == 1.0
                ), "population fractions must sum to 1"
        if hasattr(self, "CONTACT_MATRIX") and hasattr(self, "NUM_AGE_GROUPS"):
            if self.CONTACT_MATRIX:
                assert self.CONTACT_MATRIX.shape == (
                    self.NUM_AGE_GROUPS,
                    self.NUM_AGE_GROUPS,
                ), "CONTACT_MATRIX must be of shape %s, received %s" % (
                    str(
                        (
                            self.NUM_AGE_GROUPS,
                            self.NUM_AGE_GROUPS,
                        )
                    ),
                    str(self.CONTACT_MATRIX.shape),
                )
        if hasattr(self, "INFECTIOUS_PERIOD"):
            assert (
                self.INFECTIOUS_PERIOD >= 0
            ), "INFECTIOUS_PERIOD can not be negative"
        if hasattr(self, "EXPOSED_TO_INFECTIOUS"):
            assert (
                self.EXPOSED_TO_INFECTIOUS >= 0
            ), "EXPOSED_TO_INFECTIOUS can not be negative"

        if hasattr(self, "STRAIN_SPECIFIC_R0"):
            assert (
                len(self.STRAIN_SPECIFIC_R0) > 0
            ), "Must specify at least 1 strain R0"

        if hasattr(self, "NUM_WANING_COMPARTMENTS"):
            assert (
                self.NUM_WANING_COMPARTMENTS >= 0
            ), "cant have negative number of waning compartments"

            assert hasattr(
                self, "WANING_TIMES"
            ), "NUM_WANING_COMPARTMENTS without their waning times is incomplete description of waning"
            assert all(
                [wane_time >= 1 for wane_time in self.WANING_TIMES[:-1]]
            ), "Can not have waning time less than 1 day, time is in days if you meant to put months"
            assert all(
                [
                    isinstance(wane_time, int)
                    for wane_time in self.WANING_TIMES[:-1]
                ]
            ), "WANING_TIME must be of type list[int], no fractional days"
            assert (
                self.WANING_TIMES[-1] == 0
            ), "Waning times must end in 0 to account for last waning compartment not waning into anything"

            if hasattr(self, "WANING_PROTECTIONS"):
                assert self.NUM_WANING_COMPARTMENTS == len(
                    self.WANING_PROTECTIONS
                ), "unable to load config, NUM_WANING_COMPARTMENTS must equal to len(WANING_PROTECTIONS)"

        if hasattr(self, "CROSSIMMUNITY_MATRIX"):
            assert (
                self.CROSSIMMUNITY_MATRIX is None
                or self.CROSSIMMUNITY_MATRIX.shape
                == (
                    self.NUM_STRAINS,
                    self.NUM_PREV_INF_HIST,
                )
            ), "CROSSIMMUNITY_MATRIX is explicitly specified in config but shape is incorrect"
        if hasattr(self, "VAX_EFF_MATRIX"):
            assert self.VAX_EFF_MATRIX.shape == (
                self.NUM_STRAINS,
                self.MAX_VAX_COUNT + 1,
            ), "Vaccine effectiveness matrix shape incorrect"
