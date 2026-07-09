"""A module for inference processes in Dynode."""

from .inference_process import InferenceProcess
from .mcmc_process import MCMCProcess
from .svi_process import SVIProcess

__all__ = ["InferenceProcess", "MCMCProcess", "SVIProcess"]
