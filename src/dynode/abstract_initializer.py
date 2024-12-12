"""
A module that creates an abstract class for an initializer object.
An initializer objects primary purpose is initialize the state on which ODEs will be run.
AbstractInitializers will often be tasked with reading, parsing, and combining data sources
to produce an initial state representing some analyzed population
"""

from abc import ABC, abstractmethod

from . import SEIC_Compartments, utils
from typing import Any

class AbstractInitializer(ABC):
    """
    An Abstract class meant for use by disease-specific initializers.
    an initializers sole responsibility is to return an INITIAL_STATE
    parameter via self.get_initial_state().
    """

    @abstractmethod
    def __init__(self, initializer_config) -> None:
        # add these for mypy
        self.INITIAL_STATE: SEIC_Compartments | None = None
        self.config : Any = {}
        pass

    def get_initial_state(
        self,
    ) -> SEIC_Compartments:
        """
        Returns the initial state of the model as defined by the child class in __init__
        """
        assert(self.INITIAL_STATE is not None)
        return self.INITIAL_STATE

    def load_initial_population_fractions(self) -> None:
        """
        a wrapper function which loads age demographics for the US and sets the inital population fraction by age bin.

        Updates
        ----------
        `self.config.INITIAL_POPULATION_FRACTIONS` : numpy.ndarray
            proportion of the total population that falls into each age group,
            length of this array is equal the number of age groups and will sum to 1.0.
        """
        populations_path = (
            self.config.DEMOGRAPHIC_DATA_PATH
            + "population_rescaled_age_distributions/"
        )
        # TODO support getting more regions than just 1
        self.config.INITIAL_POPULATION_FRACTIONS = utils.load_age_demographics(
            populations_path, self.config.REGIONS, self.config.AGE_LIMITS
        )[self.config.REGIONS[0]]
