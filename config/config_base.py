from enum import IntEnum

import jax.numpy as jnp

base_parameters = {}


class ConfigBase:
    """
    This is the base config file, it will define a basic, runnable, set of reasonable parameters on which a model can be fit and run.
    Models that wish to test particular scenarios, modifying R0 or any other parameters, can inherit the values from config_base (this)
    and then override the needed parameters to test their scenario. These config_scenario object files can also introduce new parameters similarly.

    FAQ: why isnt this config file in YAML or JSON?
         Because these config files define complex curves (such as waning curves) that depend on the values of other parameters
         Neither YAML nor JSON supports both dynamic expressions and calculations, thus we mirror their datastructures via classes.
         This gives us the same benefit of inheritance, meaning less code duplication across config files.
    """

    def __init__(self, **kwargs) -> None:
        # fill in default parameters, may be later overriden by kwargs
        self.SCENARIO_NAME = "Base Scenario"
        self.REGIONS = ["United States"]
        self.DEMOGRAPHIC_DATA = "data/demographic-data/"
        self.SEROLOGICAL_DATA = "data/serological-data/"
        self.SAVE_PATH = "../output/"
        # CONTACT MATRICES & DEMOGRAPHY
        self.MINIMUM_AGE = 0  # why was this 1
        # age limits for each age bin in the model, begining with minimum age
        # values are exclusive in upper bound. so [0,18) means 0-17, 18+
        self.AGE_LIMITS = [self.MINIMUM_AGE, 18, 50, 65]
        self.NUM_STRAINS = 3
        # FIXED SEIR PARAMETERS
        self.POP_SIZE = 20000
        self.BIRTH_RATE = 1 / 75.0  # mu #TODO IMPLEMENT DEATHS
        # informed by source 5 (see bottom of file)
        self.INFECTIOUS_PERIOD = 7.0  # gamma
        # informed by mean of Binom(0.53, gamma(3.1, 1.6)) + 1, sources 4 and 5 (see bottom of file)
        self.EXPOSED_TO_INFECTIOUS = 3.6  # sigma
        self.VACCINATION_RATE = 1 / 500.0  # vac_p
        self.INITIAL_INFECTIONS = 1.0
        self.STRAIN_SPECIFIC_R0 = jnp.array([1.5, 1.5, 1.5])  # R0s
        self.NUM_WANING_COMPARTMENTS = 18
        self.WANING_TIME = 21  # time in WHOLE days before a recovered individual moves to first waned compartment
        self.INITAL_PROTECTION = (
            0.52  # %likelihood of re-infection given just recovered source 17
        )
        # protection against infection in each stage of waning, influenced by source 20
        # setting the following to None will get the model to initalize them from demographic/serological data
        self.INITIAL_POPULATION_FRACTIONS = None
        self.CONTACT_MATRIX = None
        self.INIT_INFECTION_DIST = None
        self.INIT_EXPOSED_DIST = None
        self.INIT_WANING_DIST = None
        self.INIT_RECOVERED_DIST = None
        self.NUM_COMPARTMENTS = 5
        # indexes ENUM for readability in code
        self.IDX = IntEnum("idx", ["S", "E", "I", "R", "W"], start=0)
        self.AXIS_IDX = IntEnum("idx", ["age", "strain", "wane"], start=0)
        # setting default rng keys
        self.MCMC_PRNGKEY = 8675309
        self.MCMC_NUM_WARMUP = 1000
        self.MCMC_NUM_SAMPLES = 1000
        self.MCMC_NUM_CHAINS = 4
        self.MCMC_PROGRESS_BAR = True
        self.MODEL_RAND_SEED = 8675309

        # now update all parameters from kwargs, overriding the defaults if they are explicitly set
        self.__dict__.update(kwargs)
        # some config params rely on other config params which may have just changed!
        # set those config params below now that everything is updated to a possible scenario.
        self.NUM_AGE_GROUPS = len(self.AGE_LIMITS)
        self.W_IDX = IntEnum(
            "w_idx",
            ["W" + str(idx) for idx in range(self.NUM_WANING_COMPARTMENTS)],
            start=0,
        )
        self.WANING_TIME_MONTHS = self.WANING_TIME / 30.0
        self.init_waning_protections_if_not_set()
        # Check that no values are incongruent with one another
        ConfigBase.assert_valid_values(self)

    def init_waning_protections_if_not_set(self):
        """
        Checks if the waning protections curve is initalized by some scenario,
        defaults to a waning protections curve as described by TODO
        """
        self.WANING_PROTECTIONS = (
            jnp.array(
                [
                    self.INITAL_PROTECTION
                    / (1 + jnp.e ** (-(2.46 - (0.2 * t))))
                    for t in jnp.linspace(
                        self.WANING_TIME_MONTHS,
                        self.WANING_TIME_MONTHS * self.NUM_WANING_COMPARTMENTS,
                        self.NUM_WANING_COMPARTMENTS,
                    )
                ]
            )
            if "WANING_PROTECTIONS" not in self.__dict__.keys()
            else self.WANING_PROTECTIONS
        )

    def assert_valid_values(self):
        assert self.POP_SIZE > 0, "population size must be a non-zero value"
        assert self.BIRTH_RATE >= 0, "BIRTH_RATE can not be negative"
        assert (
            self.INFECTIOUS_PERIOD >= 0
        ), "INFECTIOUS_PERIOD can not be negative"
        assert (
            self.EXPOSED_TO_INFECTIOUS >= 0
        ), "EXPOSED_TO_INFECTIOUS can not be negative"
        assert (
            self.VACCINATION_RATE >= 0
        ), "EXPOSED_TO_INFECTIOUS can not be negative"
        assert (
            self.INITIAL_INFECTIONS >= 0
        ), "INITIAL_INFECTIONS can not be negative"
        assert (
            len(self.STRAIN_SPECIFIC_R0) > 0
        ), "Must specify at least 1 strain R0"
        assert (
            len(self.STRAIN_SPECIFIC_R0) == self.NUM_STRAINS
        ), "Number of R0s must match number of strains"
        assert self.NUM_WANING_COMPARTMENTS == len(
            self.WANING_PROTECTIONS
        ), "unable to load config, NUM_WANING_COMPARTMENTS must equal to len(WANING_PROTECTIONS)"
        assert self.NUM_AGE_GROUPS == len(
            self.AGE_LIMITS
        ), "Number of age bins must match the NUM_AGE_GROUPS variable"
        assert (
            len(self.REGIONS) == 1
        ), "Currently model can only run on one Region at a time"

    def __str__(self):
        return str(self.__dict__)


"""
SOURCES:
Contact Matrices sourced from: https://github.com/mobs-lab/mixing-patterns

4) L. C. Tindale, et al., eLife 9, e57149 (2020). Publisher: eLife Sciences Publications, Ltd.

5) National Centre for Infectious Disease, Academy of Medicine, Singapore, Position Statement from the National Centre for Infectious Diseases and the Chapter of Infectious Disease
Physicians, Academy of Medicine, Singapore: Period of Infectivity to Inform Strategies for
De-isolation for COVID-19 Patients. (2020).

17)  D. S. Khoury, et al., Nature Medicine 27, 1205 (2021).

20)  S. Y. Tartof, et al., The Lancet 398, 1407 (2021).
"""
