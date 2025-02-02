"""DynODE, a dynamic ordinary differential model framework.

DynODE is a a compartmental mechanistic ODE model that accounts for
age structure, immunity history, vaccination, immunity waning and
multiple variants.

DynODE is currently under active development and will be substantially
refactored in the near future!
"""

# ruff: noqa: E402
import jax

"""
SEIC Compartments defines a tuple of the four major compartments used in the model
S: Susceptible, E: exposed, I: Infectious, C: cumulative (book keeping)
the dimension definitions of each of these compartments is
defined by the following Enums within the global configuration file
S: S_AXIS_IDX
E/I/C: I_AXIS_IDX
The exact sizes of each of these dimensions also depend on the implementation and config file
"""
SEIC_Compartments = tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]
# a timeseries is a tuple of compartment sizes where the leading dimension is time
# so SEIC_Timeseries has shape (tf, SEIC_Compartments.shape) for some number of timesteps tf
SEIC_Timeseries = tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]

from . import utils, vis_utils

# keep imports relative to avoid circular importing
from .abstract_initializer import AbstractInitializer
from .abstract_parameters import AbstractParameters
from .config import Config
from .covid_sero_initializer import CovidSeroInitializer
from .dynode_runner import AbstractDynodeRunner
from .mechanistic_inferer import MechanisticInferer
from .mechanistic_runner import MechanisticRunner
from .static_value_parameters import StaticValueParameters

# Defines all the different modules able to be imported from src
__all__ = [
    "AbstractParameters",
    "AbstractInitializer",
    "CovidSeroInitializer",
    "MechanisticInferer",
    "MechanisticRunner",
    "StaticValueParameters",
    "utils",
    "Config",
    "vis_utils",
    "AbstractDynodeRunner",
]
