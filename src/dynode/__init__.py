"""DynODE, a dynamic ordinary differential model framework.

DynODE is a compartmental mechanistic ODE model that accounts for
age structure, immunity history, vaccination, immunity waning and
multiple variants.

DynODE is currently under active development and will be substantially
refactored in the near future!
"""

import importlib

from . import config, infer, simulate, typing, utils

# Defines all the different modules able to be imported from src
__all__ = ["config", "infer", "utils", "simulate", "typing"]
submodules = ["config", "infer", "utils", "simulate", "typing"]
# Append the __all__ of all submodules to the main __all__
for submodule in submodules:
    module = importlib.import_module(f".{submodule}", package="dynode")
    if hasattr(module, "__all__"):
        for attr in module.__all__:
            globals()[attr] = getattr(module, attr)
            __all__.append(attr)
# effictively flattens all submodules into dynode namespace.
