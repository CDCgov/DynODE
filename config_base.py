"""
Lots of this config is sourced from Trevor's config in cdcent/cfa-mechanistic-python 
and will be adapted as needed to be used for a covid model
"""
import numpy as np
import jax.numpy as jnp
from enum import IntEnum

REGIONS = [
    "United States"
]
SEASONS = sorted([
    "22-23"
]) 
FORECAST_TARGET_DATE = "2022-11-21" # ISO format
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
    MIXING_PATH = "../data/demographic-data/"
    FLUSION_PATH = "../data/flusion-data/"
    SAVE_PATH = "../assets/figures/"

    # CONTACT MATRICES & DEMOGRAPHY
    MINIMUM_AGE = 0
    AGE_LIMITS = [4, 17, 49, 64] 
    NUM_AGE_GROUPS = len(AGE_LIMITS) + 1
    NUM_SUBTYPES = 3
    AGE_GROUPS = [
        ["0-4 yr", "5-17 yr"],
        ["18-29 yr", "30-39 yr", "40-49 yr", "50-64 yr"],
        ["65-74 yr", "75-84 yr", "85+"],
    ]
    
    # PROPERTIES OF DATA
    SEASON_FIRST_MONTH = 8
    SEASON_FIRST_WEEK = 8
    SEASON_FIRST_DAY = 1
    MAX_DAYS_AHEAD = 112 #365 # 365 is default
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
    USE_NO_POOLING = True
    USE_PARTIAL_POOLING = False
    USE_SINGLE_STATE = False
    assert (USE_NO_POOLING + USE_PARTIAL_POOLING + USE_SINGLE_STATE) == 1,"You must select one form of model to run"

    # FIXED SEIR PARAMETERS
    SPLINE_DEGREES_OF_FREEDOM = 3
    SPLINE_POINTS = np.arange(1.0, 401.0) 
    MORALITY_RATE = 1 / 75.0 # mu
    RECOVERY_RATE = 365.0 # gamma
    EXTERNAL_FORCING = 1e-3 # chi_ref
    STRAIN_INCUBATION_RATES = np.array([1 / 1.9, 1 / 1.9, 1 / 1.6]) * 365 # alpha_s
    VACCINE_PARAMETERS = [
        [17.0796, 0.55319, 5.76992, 0.587439],
        [22.2553, 0.340217, 6.45388, 0.427256],
        [23.5317, 0.625464, 4.26573, 0.767935],
    ] # vac_p
    INIT_VACCINE_PROPORTIONS = [0.0247099, 0.00548996, 0.0113005] #V_0
    VACCINE_SWITCH_POINT = 0.25 #t_1, 
    HOSPITALIZATION_RATE = (0.01 / 0.5) * (1.44 / 100) # delta_as
    ENDING_SUSCEPTIBILITY = jnp.full((3, 3), 0.5)

    # INFERABLE PARAMETER PRIORS
    SUBTYPE_SPECIFIC_R0 = [1.3, 1.6, 1.1] # R0s
    SUBTYPE_SPECIFIC_INFECTION_RATE = [1.0, 1.0, 1.0]  # chi
    RELATIVE_SCHOOL_INFECTIOUSNESS = 2 # sch_scale
    SUBTYPE_AGE_HOSPITALIZATION_RATE = "" # delta_as
    SPLINE_BASIS_PARAMETER = np.full(365, 1.3) # b
    INITIAL_SUSCEPTIBLES = np.ones((3, 3)) # initS
    VACCINE_EFFECTIVENESS = 0.55 #v_eff
    OVERALL_VACCINE_EFFECIVENESS = 0.60 # v_eff_ovr
    PROPORTION_VACCINES_EFFECTIVE = 0.35 #f_p
    FLU_TEST_RATE = 0.5 # ? 
    DELAY = "" # Z_delay
    HOSPITALIZATION_RATE_UNCERTAINTY = 0.5 #sigma_hosp

    # DIFFRAX ODE SOLVER OPTIONS
    STEP_SIZE_FIRST_STEP = 0.05 # dt0
    # OPTION: SUB_SAVE_AT 

    #compartment indexes for readability in code
    NUM_COMPARTMENTS = 6
    #todo figure out IntEnum
    idx = IntEnum('idx', ['S', 'E', 'I', 'R', 'W', 'V'], start=0)

    
    
    

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# INFERENCE
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


class InferenceConfig:

    SAMPLE_COUNT = 10000.0
    DIFFERENTIAL_EQ_SOLVER = "TSIT5"
    MCMC_PRNGKEY = 43
    MCMC_NUM_WARMUP = 500
    MCMC_NUM_SAMPLES = 2500
    MCMC_NUM_CHAINS = 4
    MCMC_PROGRESS_BAR = False
    MODEL_RAND_SEED = 324
