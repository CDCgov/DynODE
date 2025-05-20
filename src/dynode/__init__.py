"""DynODE, a dynamic ordinary differential model framework.

DynODE is a compartmental mechanistic ODE model that accounts for
age structure, immunity history, vaccination, immunity waning and
multiple variants.

DynODE is currently under active development and will be substantially
refactored in the near future!
"""

from . import config, infer, simulate, typing, utils

# Defines all the different modules able to be imported from src
__all__ = ["config", "infer", "utils", "simulate", "typing"]
