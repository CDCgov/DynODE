import datetime
import os
import subprocess
from enum import IntEnum

import git
import jax
import jax.numpy as jnp
from jax.scipy.stats.norm import pdf


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
        self.SIM_DATA = "data/abm_population.csv"
        self.VAX_MODEL_DATA = "data/spline_fits.csv"
        self.SAVE_PATH = "output/"
        # model initialization date DO NOT CHANGE
        self.INIT_DATE = datetime.date(2022, 2, 11)
        self.MINIMUM_AGE = 0
        # age limits for each age bin in the model, begining with minimum age
        # values are exclusive in upper bound. so [0,18) means 0-17, 18+
        self.AGE_LIMITS = [self.MINIMUM_AGE, 18, 50, 65]
        # Total Population size of simulation
        self.POP_SIZE = 10000
        # Time in days an individual is infectious for informed by source 5 (see bottom of file)
        self.INFECTIOUS_PERIOD = 7.0  # gamma
        # time in days between exposure to virus to infectious and able to pass to others
        # informed by mean of Binom(0.53, gamma(3.1, 1.6)) + 1, sources 4 and 5 (see bottom of file)
        self.EXPOSED_TO_INFECTIOUS = 3.6  # sigma
        # rate of vaccinations per num individuals vaccinated each day?
        # TODO informed by?
        self.VACCINATION_RATE = 1 / 500.0  # vac_p
        # number of vaccines maximum for an individual, any more are not counted with bonus immunity.
        self.MAX_VAX_COUNT = 2
        # Initial Infections in the model, these are dispersed between exposed and infectious states
        # If None, sourced via the proportion of infections from Tom's ABM * POP_SIZE
        self.INITIAL_INFECTIONS = None
        # R0 values of each strain, from oldest to newest. Including R0 for introduced strains
        self.STRAIN_SPECIFIC_R0 = jnp.array([1.2, 1.8, 3.0])  # R0s
        # days after model initialization when new strains are externally introduced
        self.INTRODUCTION_TIMES = [60]
        # the percentage of the total population as a float who are externally introduced with the new strain.
        self.INTRODUCTION_PERCENTAGE = 0.01
        # mask of what age bins to introduce external infected populations as.
        # with 4 age bins, 0-17, 18-49, 50-64, 65+ a True in the first index corresponds to 0-17 aged infected introduced
        self.INTRODUCTION_AGE_MASK = [False, True, False, False]
        # number of compartments individuals wane through the moment of recovery.
        # there is no explicit "recovered" compartment.
        self.NUM_WANING_COMPARTMENTS = 4
        # the % protection from reinfection offered to individuals in each waning compartment.
        # TODO SOURCE?
        self.WANING_PROTECTIONS = jnp.array([1.0, 0.985, 0.985, 0])
        # WANING_TIMES in days for each waning compartment, ends in 0 as last compartment does not wane
        self.WANING_TIMES = [21, 142, 142, 0]
        # TODO use priors informed by https://www.sciencedirect.com/science/article/pii/S2352396423002992
        # the protection afforded by different immune histories from infection.
        # non-omicron vs omicron, stratified by immune history. 0 = fully susceptible, 1 = fully immune.
        # TODO SOURCE?
        self.STRAIN_INTERACTIONS = jnp.array(
            [
                [1.0, 0.7, 0.49],  # delta
                [0.7, 1.0, 0.7],  # omi
                [0.49, 0.7, 1.0],  # BA1.1
            ]
        )
        # TODO fix this to use SxS matrix to inform this.
        # self.CROSSIMMUNITY_MATRIX = jnp.array(
        #     [
        #         [0, 0.5, 0.7, 0.8, 0.8, 0.8, 0.8, 0.8],  # delta
        #         [0, 0.3, 0.5, 0.7, 0.8, 0.8, 0.8, 0.8],  # omicron
        #         [0, 0.1, 0.3, 0.5, 0.7, 0.7, 0.8, 0.8],  # BA1.1
        #     ]
        # )  # TODO what do we do about delta challenging someone with prev omi2 exposure?
        # the protection afforded by different numbers of vaccinations from infection.
        # non-omicron vs omicron, stratified by vaccine count, 0, 1, 2+ shots. 0 = fully susceptible, 1 = fully immune.
        # TODO SOURCE?
        self.VAX_EFF_MATRIX = jnp.array(
            [
                [0, 0.34, 0.68],  # delta
                [0, 0.24, 0.48],  # omicron1
                [0, 0.14, 0.28],  # BA1.1
            ]
        )
        # setting the following to None will get the model to initialize them from demographic/abm data
        self.INITIAL_POPULATION_FRACTIONS = None
        self.CROSSIMMUNITY_MATRIX = None
        self.CONTACT_MATRIX = None
        self.INIT_INFECTION_DIST = None
        self.INIT_EXPOSED_DIST = None
        self.INIT_IMMUNE_HISTORY = None
        self.INIT_INFECTED_DIST = None
        self.VAX_MODEL_PARAMS = None
        # indexes ENUM for readability in code
        self.IDX = IntEnum("idx", ["S", "E", "I", "C"], start=0)
        self.S_AXIS_IDX = IntEnum(
            "idx", ["age", "hist", "vax", "wane"], start=0
        )
        self.I_AXIS_IDX = IntEnum(
            "idx", ["age", "hist", "vax", "strain"], start=0
        )
        # setting default rng keys
        self.MCMC_PRNGKEY = 8675309
        self.MCMC_NUM_WARMUP = 100
        self.MCMC_NUM_SAMPLES = 1000
        self.MCMC_NUM_CHAINS = 4
        self.MCMC_PROGRESS_BAR = True
        self.MODEL_RAND_SEED = 8675309

        # now update all parameters from kwargs, overriding the defaults if they are explicitly set
        self.__dict__.update(kwargs)
        self.GIT_HASH = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("ascii")
            .strip()
        )
        self.GIT_REPO = git.Repo()
        # some config params rely on other config params which may have just changed!
        # set those config params below now that everything is updated to a possible scenario.
        self.NUM_AGE_GROUPS = len(self.AGE_LIMITS)
        self.AGE_GROUP_STRS = [
            str(self.AGE_LIMITS[i - 1]) + "-" + str(self.AGE_LIMITS[i] - 1)
            for i in range(1, len(self.AGE_LIMITS))
        ] + [str(self.AGE_LIMITS[-1]) + "+"]
        self.AGE_GROUP_IDX = IntEnum("age", self.AGE_GROUP_STRS, start=0)

        self.NUM_STRAINS = len(self.STRAIN_SPECIFIC_R0)

        # enum for marking waning indexes, improving readability
        self.W_IDX = IntEnum(
            "w_idx",
            ["W" + str(idx) for idx in range(self.NUM_WANING_COMPARTMENTS)],
            start=0,
        )

        # this are all the strains currently supported, historical and future
        all_strains = [
            "wildtype",
            "alpha",
            "delta",
            "omicron",
            "BA1.1",
        ]
        # it often does not make sense to differentiate between wildtype and alpha, so combine strains here
        self.STRAIN_NAMES = all_strains[5 - self.NUM_STRAINS :]
        # in each compartment that is strain stratified we use strain indexes to improve readability.
        # omicron will always be index=2 if num_strains >= 3. In a two strain model we must combine alpha and delta together.
        self.STRAIN_IDX = IntEnum(
            "strain_idx",
            self.STRAIN_NAMES,
            start=0,
        )

        # we need some functions to model external infected persons interacting with the population
        # these functions should take some time in days
        # summing over all days should equal 1.0. While the MechanisticModel itself decides the total
        # number of external infected individuals over all timepoints.
        def zero_function(_):
            return 0

        self.EXTERNAL_I_DISTRIBUTIONS = [
            zero_function for _ in range(self.NUM_STRAINS)
        ]
        for introduced_strain_idx, introduced_time in enumerate(
            self.INTRODUCTION_TIMES
        ):
            dist_idx = self.NUM_STRAINS - introduced_strain_idx - 1
            # use a normal PDF with std dv
            self.EXTERNAL_I_DISTRIBUTIONS[dist_idx] = lambda t: pdf(
                t, loc=introduced_time, scale=7
            )

        # Vaccination modeling, using cubic splines to model vax uptake in the population stratified by age and current vax shot.
        def base_equation(t, coefficients):
            """
            the base of a spline equation, without knots, follows a simple cubic formula
            a + bt + ct^2 + dt^3. This is a vectorized version of this equation which takes in
            a matrix of `a` values, as well as a marix of `b`, `c`, and `d` coefficients.
            PARAMETERS
            ----------
            t: jax.tracer array
                a jax tracer containing within it the time in days since model simulation start
            intercepts: jnp.array()
                intercepts of each cubic spline base equation for all combinations of age bin and vax history
                intercepts.shape=(NUM_AGE_GROUPS, MAX_VAX_COUNT + 1)
            coefficients: jnp.array()
                coefficients of each cubic spline base equation for all combinations of age bin and vax history
                coefficients.shape=(NUM_AGE_GROUPS, MAX_VAX_COUNT + 1, 3)
            """
            # we are taking the derivative of the base equation, so t^2 becomes 2t
            # and t^3 becomes 3t^2, drop the intercept
            return jnp.sum(
                coefficients
                * jnp.array([0, 1, 2 * t, 3 * t**2])[
                    jnp.newaxis, jnp.newaxis, :
                ],
                axis=-1,
            )

        def conditional_knots(t, knots, coefficients):
            indicators = jnp.where(t > knots, t - knots, 0)
            # multiply coefficients by 3 since we taking derivative of cubic spline.
            return jnp.sum(indicators**2 * (3 * coefficients), axis=-1)

        # days of separation between each knot
        self.VAX_MODEL_KNOT_SEPARATION = 7
        self.VAX_MODEL_KNOTS = jnp.array(
            list(
                range(
                    self.VAX_MODEL_KNOT_SEPARATION,
                    449,
                    self.VAX_MODEL_KNOT_SEPARATION,
                )
            )
        )

        def VAX_FUNCTION(t, knots, coefficients):
            """
            Returns the value of a cubic spline with knots and coefficients evaluated on day `t` for each age_bin x vax history combination.
            Cubic spline equation f(t) = a + bt + ct^2 + dt^3 + sum_i_len(knots) {coef[i] * (t-knots[i])^3 * I(t > knots[i]) }

            Where coef/knots[i] is the i'th index of each array. and the I() function is an indicator variable 1 or 0.

            PARAMETERS
            ----------
            t: jax.tracer array
                a jax tracer containing within it the time in days since model simulation start
            knots: jnp.array()
                knot locations of each cubic spline for all combinations of age bin and vax history
                knots.shape=(NUM_AGE_GROUPS, MAX_VAX_COUNT + 1, # knots in each spline)
            coefficients: jnp.array()
                knot coefficients of each cubic spline for all combinations of age bin and vax history.
                including first 4 coefficients for the base equation.
                coefficients.shape=(NUM_AGE_GROUPS, MAX_VAX_COUNT + 1, # knots in each spline + 4)

            Returns
            ----------
            jnp.array() containing the proportion of individuals in each age x vax combination that will be vaccinated during this time step.
            """
            base = base_equation(t, coefficients.at[:, :, 0:4].get())
            knots = conditional_knots(
                t, knots, coefficients.at[:, :, 4:].get()
            )
            return jax.nn.relu(base + knots)

        self.VAX_FUNCTION = VAX_FUNCTION

        # times at which the beta value in transmission dynamics may need to be adjusted
        self.BETA_TIMES = jnp.array([0.0, 120.0, 150])
        # coefficients that the beta value will be multiplied with at time t in BETA_TIMES
        self.BETA_COEFICIENTS = jnp.array([1.0, 1.0, 1.0])
        # number of previous infection histories depends on the number of strains being tested.
        # can be either infected or not infected by each strain.
        self.NUM_PREV_INF_HIST = 2**self.NUM_STRAINS
        # Check that no values are incongruent with one another
        self.assert_valid_values()

    def assert_valid_values(self):
        """
        a function designed to be called after all parameters are initialized, does a series of reasonable checks
        to ensure values are within expected ranges and no parameters directly contradict eachother.

        Raises
        ----------
        Assert Error:
            if user supplies invalid parameters, short description will be provided as to why the parameter is wrong.
        """
        assert os.path.exists(self.SAVE_PATH), (
            "%s is not a valid path" % self.SAVE_PATH
        )
        assert os.path.exists(self.DEMOGRAPHIC_DATA), (
            "%s is not a valid path" % self.DEMOGRAPHIC_DATA
        )
        assert os.path.exists(self.SEROLOGICAL_DATA), (
            "%s is not a valid path" % self.SEROLOGICAL_DATA
        )
        assert os.path.exists(self.SIM_DATA), (
            "%s is not a valid path" % self.SIM_DATA
        )
        assert os.path.exists(self.VAX_MODEL_DATA), (
            "%s is not a valid path" % self.VAX_MODEL_DATA
        )
        assert self.MINIMUM_AGE >= 0, "no negative minimum ages, lowest is 0"
        assert (
            self.AGE_LIMITS[0] == self.MINIMUM_AGE
        ), "first age in AGE_LIMITS must be self.MINIMUM_AGE"
        assert all(
            [
                self.AGE_LIMITS[idx] > self.AGE_LIMITS[idx - 1]
                for idx in range(1, len(self.AGE_LIMITS))
            ]
        ), "AGE_LIMITS must be strictly increasing"
        assert (
            self.AGE_LIMITS[-1] < 85
        ), "age limits can not exceed 84 years of age, the last age bin is implied and does not need to be included"
        assert self.POP_SIZE > 0, "population size must be a non-zero value"
        if self.INITIAL_INFECTIONS:
            assert (
                self.INITIAL_INFECTIONS <= self.POP_SIZE
            ), "cant have more initial infections than total population size"

            assert (
                self.INITIAL_INFECTIONS >= 0
            ), "cant have negative initial infections"

        # if user has supplied custom values for distributions instead of using prebuilt ones, sanity check them here
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

        if self.INIT_INFECTION_DIST:
            assert self.INIT_INFECTION_DIST.shape == (
                self.NUM_AGE_GROUPS,
            ), "INIT_INFECTION_DIST must be of shape %s, received %s" % (
                str((self.NUM_AGE_GROUPS,)),
                str(self.INIT_INFECTION_DIST.shape),
            )

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
        if self.INIT_INFECTED_DIST:
            assert self.INIT_INFECTED_DIST.shape == (
                self.NUM_AGE_GROUPS,
                self.NUM_PREV_INF_HIST,
                self.MAX_VAX_COUNT + 1,
                self.NUM_STRAINS,
            ), "INIT_INFECTED_DIST must be of shape %s, received %s" % (
                str(
                    (
                        self.NUM_AGE_GROUPS,
                        self.NUM_PREV_INF_HIST,
                        self.MAX_VAX_COUNT + 1,
                        self.NUM_STRAINS,
                    )
                ),
                str(self.INIT_INFECTED_DIST.shape),
            )

        if self.INIT_EXPOSED_DIST:
            assert self.INIT_EXPOSED_DIST.shape == (
                self.NUM_AGE_GROUPS,
                self.NUM_PREV_INF_HIST,
                self.MAX_VAX_COUNT + 1,
                self.NUM_STRAINS,
            ), "INIT_EXPOSED_DIST must be of shape %s, received %s" % (
                str(
                    (
                        self.NUM_AGE_GROUPS,
                        self.NUM_PREV_INF_HIST,
                        self.MAX_VAX_COUNT + 1,
                        self.NUM_STRAINS,
                    )
                ),
                str(self.INIT_EXPOSED_DIST.shape),
            )

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
            len(self.STRAIN_SPECIFIC_R0) > 0
        ), "Must specify at least 1 strain R0"
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
        assert self.NUM_STRAINS >= 1, "No such thing as a 0 strain model"
        assert (
            len(self.STRAIN_SPECIFIC_R0) == self.NUM_STRAINS
        ), "Number of R0s must match number of strains"
        assert (
            self.NUM_WANING_COMPARTMENTS >= 0
        ), "cant have negative number of waning compartments"
        assert self.NUM_WANING_COMPARTMENTS == len(
            self.WANING_PROTECTIONS
        ), "unable to load config, NUM_WANING_COMPARTMENTS must equal to len(WANING_PROTECTIONS)"
        assert (
            self.CROSSIMMUNITY_MATRIX is None
            or self.CROSSIMMUNITY_MATRIX.shape
            == (
                self.NUM_STRAINS,
                self.NUM_PREV_INF_HIST,
            )
        ), "CROSSIMMUNITY_MATRIX shape incorrect"
        assert self.VAX_EFF_MATRIX.shape == (
            self.NUM_STRAINS,
            self.MAX_VAX_COUNT + 1,
        ), "Vaccine effectiveness matrix shape incorrect"
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
