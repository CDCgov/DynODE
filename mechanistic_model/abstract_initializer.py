"""
The following abstract class defines a mechanistic_initializer,
which is used by mechanistic_runner instances to initialize the state on which ODEs will be run.
mechanistic_initializers will often be tasked with reading, parsing, and combining data sources
to produce an initial state representing some analyzed population
"""

from abc import ABC, abstractmethod

import utils


class MechanisticInitializer(ABC):
    @abstractmethod
    def __init__(self, initializer_config):
        self.INITIAL_STATE
        pass

    def get_initial_state(self):
        """
        Returns the initial state of the model as defined by the child class in __init__
        """
        return self.INITIAL_STATE

    def load_initial_population_fractions(self):
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
