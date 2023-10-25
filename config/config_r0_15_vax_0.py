"""
Lots of this config is sourced from Trevor's config in cdcent/cfa-mechanistic-python 
and will be adapted as needed to be used for a covid model
"""
import numpy as np
import jax.numpy as jnp
from enum import IntEnum

REGIONS = ["United States"]
SEASONS = sorted(["22-23"])
FORECAST_TARGET_DATE = "2022-11-21"  # ISO format
FORECAST_HORIZON = 4


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# DATA LOADING
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


class DataConfig:
    # RELATIVE PATHS
    NREVSS_PATH = "../data/nrevss-data/"
    HHS_PROTECT_PATH = "../data/hhs-data/"
    MIXING_PATH = "data/demographic-data/"
    FLUSION_PATH = "../data/flusion-data/"
    SAVE_PATH = "../assets/figures/"

    # CONTACT MATRICES & DEMOGRAPHY
    MINIMUM_AGE = 0  # why was this 1
    # values are exclusive in upper bound. so [0,18) means 0-17, 18+
    AGE_LIMITS = [MINIMUM_AGE, 18, 50, 65]
    NUM_AGE_GROUPS = len(AGE_LIMITS)
    NUM_STRAINS = 3
    AGE_GROUPS = [
        ["0-4 yr", "5-17 yr"],
        ["18-29 yr", "30-39 yr", "40-49 yr", "50-64 yr"],
        ["65-74 yr", "75-84 yr", "85+"],
    ]
    # PROPERTIES OF DATA
    SEASON_FIRST_MONTH = 8
    SEASON_FIRST_WEEK = 8
    SEASON_FIRST_DAY = 1
    MAX_DAYS_AHEAD = 112  # 365 # 365 is default
    # OPTION: SEASON_LAST_MONTH
    # OPTION: SEASON_LAST_WEEK
    # OPTION: SEASON_LAST_DAY

    # FURTHER DATA SPECIFICATIONS
    FIT_JUNE_AND_JULY = True
    NORMALIZE_FLUSION = True
    USE_NREVSS_CLINICAL_LAB = True
    USE_NREVSS_PUBLIC_HEALTH = True
    USE_FLUSION = True
    USE_HHS = True
    USE_CONTACT_MATRICES = True

    # INFORMATION LEVELS
    VERBOSE_OUTPUT = False
    # OPTION: USE_NRVESS_CLINICAL_LAB
    # OPTION: USE_NRVESS_PUBLIC_HEALTH
    # OPTION: VIS_NRVESS_PUBLIC_HEALTH
    # OPTION: VIS_NRVESS_CLINICAL_LAB
    # OPTION: VIS_HHS_PROTECT
    # OPTION: VIS_FLUSION
    # OPTION: VIS_MIXING_MATRICES
    # OPTION: VIS_POPULATION_PROPORTIONS
    # OPTION: VIS_POPULATION
    # OPTION: SAVE_VIS_NRVESS_PUBLIC_HEALTH
    # OPTION: SAVE_VIS_NRVESS_CLINICAL_LAB
    # OPTION: SAVE_VIS_HHS_PROTECT
    # OPTION: SAVE_VIS_FLUSION
    # OPTION: SAVE_VIS_MIXING_MATRICES
    # OPTION: SAVE_VIS_POPULATION_PROPORTIONS
    # OPTION: SAVE_VIS_POPULATION


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# MODEL TYPE
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


class ModelConfig:
    # the model age bins / strains must match the data being read in
    NUM_AGE_GROUPS = DataConfig.NUM_AGE_GROUPS
    NUM_STRAINS = DataConfig.NUM_STRAINS
    AGE_LIMITS = DataConfig.AGE_LIMITS
    # FIXED SEIR PARAMETERS
    POP_SIZE = 20000
    assert POP_SIZE > 0, "population size must be a non-zero value"

    BIRTH_RATE = 1 / 75.0  # mu #TODO IMPLEMENT DEATHS
    assert BIRTH_RATE >= 0, "BIRTH_RATE can not be negative"

    # informed by source 5 (see bottom of file)
    INFECTIOUS_PERIOD = 7.0  # gamma
    assert INFECTIOUS_PERIOD >= 0, "INFECTIOUS_PERIOD can not be negative"

    # informed by mean of Binom(0.53, gamma(3.1, 1.6)) + 1, sources 4 and 5 (see bottom of file)
    EXPOSED_TO_INFECTIOUS = 3.6  # sigma
    assert EXPOSED_TO_INFECTIOUS >= 0, "EXPOSED_TO_INFECTIOUS can not be negative"

    VACCINATION_RATE = 0.0  # vac_p
    assert VACCINATION_RATE >= 0, "EXPOSED_TO_INFECTIOUS can not be negative"

    INITIAL_INFECTIONS = 1.0
    assert INITIAL_INFECTIONS >= 0, "INITIAL_INFECTIONS can not be negative"

    STRAIN_SPECIFIC_R0 = jnp.array([1.5, 1.5, 1.5])  # R0s
    assert len(STRAIN_SPECIFIC_R0) > 0, "Must specify at least 1 strain R0"
    assert (
        len(STRAIN_SPECIFIC_R0) == NUM_STRAINS
    ), "Number of R0s must match number of strains"

    NUM_WANING_COMPARTMENTS = 18
    WANING_TIME = 21  # time in days before a recovered individual moves to first waned compartment
    WANING_TIME_MONTHS = WANING_TIME / 30.0

    # protection against infection in each stage of waning, influenced by source 20
    WANING_PROTECTIONS = jnp.array(
        [
            0.52 / (1 + np.e ** (-(2.46 - (0.2 * t))))  # init_protection=0.52 source 17
            for t in np.linspace(
                WANING_TIME_MONTHS,
                WANING_TIME_MONTHS * NUM_WANING_COMPARTMENTS,
                NUM_WANING_COMPARTMENTS,
            )
        ]
    )
    # WANING_PROTECTIONS = jnp.array([0.88, 0.84, 0.77, 0.70, 0.61, 0.51])
    assert NUM_WANING_COMPARTMENTS == len(
        WANING_PROTECTIONS
    ), "unable to load config, NUM_WANING_COMPARTMENTS must equal to len(WANING_PROTECTIONS)"

    NUM_COMPARTMENTS = 5
    # compartment indexes ENUM for readability in code
    w_idx = IntEnum(
        "w_idx", ["W" + str(idx) for idx in range(NUM_WANING_COMPARTMENTS)], start=0
    )
    idx = IntEnum("idx", ["S", "E", "I", "R", "W"], start=0)
    axis_idx = IntEnum("idx", ["age", "strain", "wane"], start=0)


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# INFERENCE
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


class InferenceConfig:
    MCMC_PRNGKEY = 8675309
    MCMC_NUM_WARMUP = 1000
    MCMC_NUM_SAMPLES = 1000
    MCMC_NUM_CHAINS = 4
    MCMC_PROGRESS_BAR = True
    MODEL_RAND_SEED = 8675309


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
