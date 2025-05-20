"""DynODE, a dynamic ordinary differential model framework.

DynODE is a compartmental mechanistic ODE model that accounts for
age structure, immunity history, vaccination, immunity waning and
multiple variants.

DynODE is currently under active development and will be substantially
refactored in the near future!
"""

from . import config, infer, simulate, typing, utils
from .config import __all__ as config_all
from .infer import __all__ as infer_all
from .simulate import __all__ as simulate_all
from .typing import __all__ as typing_all
from .utils import __all__ as utils_all

# Defines all the different modules able to be imported from src
__all__ = ["config", "infer", "utils", "simulate", "typing"]
# Append the __all__ of all submodules to the main __all__

__all__.extend(config_all)
__all__.extend(infer_all)
__all__.extend(simulate_all)
__all__.extend(typing_all)
__all__.extend(utils_all)
