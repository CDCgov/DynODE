from typing import List

from numpyro.distributions import Distribution
from pydantic import BaseModel, ConfigDict, NonNegativeFloat, PositiveFloat
from typing_extensions import Self

from .strains import Strain


class SolverParams(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    ode_solver_rel_tolerance: PositiveFloat
    ode_solver_abs_tolerance: PositiveFloat


class TransmissionParams(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    strain_interactions: dict[str, dict[str, NonNegativeFloat | Distribution]]
    strains: List[Strain]


class InferenceParams(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)


class MCMCParams(InferenceParams):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    inference_mcmc_steps: PositiveFloat


class SVIParams(InferenceParams):
    model_config = ConfigDict(arbitrary_types_allowed=True)


class Params(BaseModel):
    """Miscellaneous parameters of an ODE model."""

    # allow users to pass custom types to ParamStore
    model_config = ConfigDict(arbitrary_types_allowed=True)
    solver_params: SolverParams
    transmission_params: TransmissionParams

    def realize_distributions(self) -> Self:
        """Go through parameters and sample all distribution objects

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
