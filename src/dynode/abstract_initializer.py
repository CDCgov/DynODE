"""A module that creates an abstract class for an initializer object.

An initializer objects primary purpose is initialize the state on which ODEs will be run.
AbstractInitializers will often be tasked with reading, parsing, and combining data sources
to produce an initial state representing some analyzed population
"""

from abc import ABC, abstractmethod
from typing import Any

from numpy import ndarray

from src.dynode.utility import log_decorator, logger

from . import utils
from .typing import SEIC_Compartments


class AbstractInitializer(ABC):
    """An abstract class meant for use by disease-specific initializers.

    An initializer's sole responsibility is to return an INITIAL_STATE
    parameter via self.get_initial_state().
    """

    @abstractmethod
    def __init__(self, initializer_config) -> None:
        """Load parameters from `initializer_config` and generate self.INITIAL_STATE.

        Parameters
        ----------
        initializer_config : str
            str path to config json holding necessary initializer parameters.
        """
        # add these for mypy
        self.INITIAL_STATE: SEIC_Compartments | None = None
        self.config: Any = {}
        pass

    @log_decorator()
    def get_initial_state(
        self,
    ) -> SEIC_Compartments:
        """Get the initial state of the model as defined by the child class in __init__.

        Returns
        -------
        SEIC_Compartments
            tuple of matricies representing initial state of each compartment
            in the model.
        """
        assert self.INITIAL_STATE is not None, logger.error(
            "INITIAL_STATE is None."
        )
        return self.INITIAL_STATE

    def load_initial_population_fractions(self) -> ndarray:
        """Load age demographics for the specified region.

        Returns
        -------
        numpy.ndarray
            Proportion of the total population that falls into each age group.
            `len(self.load_initial_population_fractions()) == self.config.NUM_AGE_GROUPS`
            `np.sum(self.load_initial_population_fractions()) == 1.0
        """
        logger.debug(
            "Creating populations_path based on DEMOGRAPHIC_DATA_PATH in config."
        )

        populations_path = (
            self.config.DEMOGRAPHIC_DATA_PATH
            + "population_rescaled_age_distributions/"
        )

        logger.debug(f"Set populations path as {populations_path}.")
        logger.debug("Returning values from utils.load_age_demographics()")

        # TODO support getting more regions than just 1
        return utils.load_age_demographics(
            populations_path, self.config.REGIONS, self.config.AGE_LIMITS
        )[self.config.REGIONS[0]]
