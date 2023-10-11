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
# OPTION: FIT_FROM_DATE =
# OPTION: FIT_UNTIL_DATE =
# OPTION: PRINT_TIMING


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
    MINIMUM_AGE = 1  # why was this 1
    AGE_LIMITS = [4, 17, 49, 64]  # TODO CHANGE TO BIN UPPER BOUNDS, BIN TO 85
    NUM_AGE_GROUPS = len(AGE_LIMITS) + 1
    NUM_STRAINS = 3
    AGE_GROUPS = [
        ["0-4 yr", "5-17 yr"],
        ["18-29 yr", "30-39 yr", "40-49 yr", "50-64 yr"],
        ["65-74 yr", "75-84 yr", "85+"],
    ]
    POP_SIZE = 20000
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
    # POOLING CONFIGURATION

    # FIXED SEIR PARAMETERS
    BIRTH_RATE = 1 / 75.0  # mu #TODO IMPLEMENT DEATHS

    assert BIRTH_RATE >= 0, "BIRTH_RATE can not be negative"
    INFECTIOUS_PERIOD = 5.0  # gamma
    assert INFECTIOUS_PERIOD >= 0, "INFECTIOUS_PERIOD can not be negative"
    EXPOSED_TO_INFECTIOUS = 2.0  # sigma
    assert EXPOSED_TO_INFECTIOUS >= 0, "EXPOSED_TO_INFECTIOUS can not be negative"
    VACCINATION_RATE = 0  # 1 / 50.0 # vac_p
    assert VACCINATION_RATE >= 0, "EXPOSED_TO_INFECTIOUS can not be negative"
    INITIAL_INFECTIONS = 1.0
    assert INITIAL_INFECTIONS >= 0, "INITIAL_INFECTIONS can not be negative"

    # INFERABLE PARAMETER PRIORS
    STRAIN_SPECIFIC_R0 = jnp.array([1.5, 1.5, 1.5])  # R0s
    assert len(STRAIN_SPECIFIC_R0) > 0, "Must specify at least 1 strain R0"
    NUM_WANING_COMPARTMENTS = 4
    # protection against infection in each stage of waning
    WANING_PROTECTIONS = jnp.array([0.7, 0.6, 0.4, 0.15])
    assert NUM_WANING_COMPARTMENTS == len(
        WANING_PROTECTIONS
    ), "unable to load config, NUM_WANING_COMPARTMENTS must equal to len(WANING_PROTECTIONS)"
    w_idx = IntEnum(
        "w_idx", ["W" + str(idx) for idx in range(NUM_WANING_COMPARTMENTS)], start=0
    )
    WANING_TIME = 20.0  # time in days before a recovered individual moves to first waned compartment
    # compartment indexes for readability in code
    NUM_COMPARTMENTS = 5
    # todo figure out IntEnum
    idx = IntEnum("idx", ["S", "E", "I", "R", "W"], start=0)


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
