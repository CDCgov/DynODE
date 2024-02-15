"""
The following abstract class defines a mechanistic_initializer,
which is used by mechanistic_runner instances to initialize the state on which ODEs will be run.
mechanistic_initializers will often be tasked with reading, parsing, and combining data sources
to produce an initial state representing some analyzed population
"""
from abc import ABC, abstractmethod


class MechanisticInitializer(ABC):
    @abstractmethod
    def __init__(self, initializer_config):
        self.INITIAL_STATE
        pass

    @abstractmethod
    def assert_valid_configuration(self):
        pass

    def get_initial_state(self):
        """
        Returns the initial state of the model as defined by the child class in __init__
        """
        return self.INITIAL_STATE
