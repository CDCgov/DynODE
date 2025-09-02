"""A module for inference processes in Dynode."""

from .checkpointing import checkpoint_compartment_sizes
from .inference import InferenceProcess, MCMCProcess, SVIProcess

__all__ = [
    "InferenceProcess",
    "MCMCProcess",
    "SVIProcess",
    "checkpoint_compartment_sizes",
]
