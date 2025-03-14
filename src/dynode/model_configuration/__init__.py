"""DynODE configuration module."""

from .config_definition import (
    Compartment,
    InferenceParams,
    Initializer,
    SimulationConfig,
)

redundant_alias_Compartment = Compartment
redundant_alias_CompartmentalModel = SimulationConfig
redundant_alias_InferenceParams = InferenceParams
redundant_alias_Initializer = Initializer
