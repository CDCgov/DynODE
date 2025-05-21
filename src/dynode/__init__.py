"""DynODE, a dynamic ordinary differential model framework.

DynODE is a compartmental mechanistic ODE model that accounts for
age structure, immunity history, vaccination, immunity waning and
multiple variants.

DynODE is currently under active development and will be substantially
refactored in the near future!
"""

from . import config, infer, simulation, typing, utils

# import everything from the submodules
from .config import (
    AgeBin,
    Bin,
    Compartment,
    DeterministicParameter,
    Dimension,
    DiscretizedPositiveIntBin,
    FullStratifiedImmuneHistoryDimension,
    Initializer,
    LastStrainImmuneHistoryDimension,
    Params,
    PlaceholderSample,
    SamplePlaceholderError,
    SimulationConfig,
    SimulationDate,
    SolverParams,
    Strain,
    TransmissionParams,
    VaccinationDimension,
    WaneBin,
)
from .infer import (
    InferenceProcess,
    MCMCProcess,
    SVIProcess,
    checkpoint_compartment_sizes,
    resolve_deterministic,
    sample_distributions,
    sample_then_resolve,
)
from .simulation import (
    AbstractODEParams,
    base_equation,
    conditional_knots,
    evaluate_cubic_spline,
    simulate,
)
from .typing import (
    CompartmentGradients,
    CompartmentState,
    CompartmentTimeseries,
    DynodeName,
    ObservedData,
    ODE_Eqns,
    UnitIntervalFloat,
)
from .utils import CustomLogFormatter, log, log_decorator, logger

# Defines all the different modules able to be imported from src
__all__ = [
    "config",
    "infer",
    "utils",
    "simulation",
    "typing",
    "SimulationConfig",
    "Initializer",
    "Compartment",
    "Strain",
    "Dimension",
    "VaccinationDimension",
    "FullStratifiedImmuneHistoryDimension",
    "LastStrainImmuneHistoryDimension",
    "Bin",
    "WaneBin",
    "DiscretizedPositiveIntBin",
    "AgeBin",
    "Params",
    "SolverParams",
    "TransmissionParams",
    "SimulationDate",
    "PlaceholderSample",
    "SamplePlaceholderError",
    "DeterministicParameter",
    "sample_then_resolve",
    "resolve_deterministic",
    "sample_distributions",
    "InferenceProcess",
    "MCMCProcess",
    "SVIProcess",
    "checkpoint_compartment_sizes",
    "simulate",
    "AbstractODEParams",
    "evaluate_cubic_spline",
    "base_equation",
    "conditional_knots",
    "CompartmentState",
    "CompartmentGradients",
    "DynodeName",
    "CompartmentTimeseries",
    "UnitIntervalFloat",
    "ObservedData",
    "ODE_Eqns",
    "log",
    "log_decorator",
    "CustomLogFormatter",
    "logger",
    "sim_day_to_date",
    "sim_day_to_epiweek",
    "date_to_sim_day",
    "date_to_epi_week",
    "vectorize_objects",
    "flatten_list_parameters",
    "drop_keys_with_substring",
    "identify_distribution_indexes",
]
