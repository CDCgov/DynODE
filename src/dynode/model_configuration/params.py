"""Module containing Parameter classes for storing DynODE parameters."""

from typing import List

from jax import Array
from jax.random import PRNGKey
from numpyro.distributions import Distribution
from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
)
from typing_extensions import Self

from .strains import Strain
from .types import DeterministicParameter


class SolverParams(BaseModel):
    """Parameters used by the ODE solver."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    ode_solver_rel_tolerance: PositiveFloat
    ode_solver_abs_tolerance: PositiveFloat


class TransmissionParams(BaseModel):
    """Transmission Parameters for the respiratory model."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    strain_interactions: dict[
        str,
        dict[str, NonNegativeFloat | Distribution | DeterministicParameter],
    ]
    strains: List[Strain]


class InferenceParams(BaseModel):
    """Parameters necessary for inference."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    inference_prngkey: Array = PRNGKey(8675314)


class MCMCParams(InferenceParams):
    """Inference parameters specific to Markov Chain Monte Carlo (MCMC) fitting methods."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    inference_mcmc_samples: PositiveInt
    inference_mcmc_warmup: PositiveInt


class SVIParams(InferenceParams):
    """Inference parameters specific to Stochastic Variational Inference (SVI) fitting methods."""

    model_config = ConfigDict(arbitrary_types_allowed=True)


class Params(BaseModel):
    """Miscellaneous parameters of an ODE model."""

    # allow users to pass custom types to ParamStore
    model_config = ConfigDict(arbitrary_types_allowed=True)
    solver_params: SolverParams
    transmission_params: TransmissionParams

    def realize_distributions(self) -> Self:
        """Go through parameters and sample all distribution objects.

        Returns
        -------
        Self
            Params with all distribution objects replaced by
            jax.Array containing samples from that distribution.

        Raises
        ------
        NotImplementedError
            Not yet implemented
        """
        raise NotImplementedError()
