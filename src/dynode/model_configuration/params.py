"""Module containing Parameter classes for storing DynODE parameters."""

from typing import List

import chex
from jax import Array
from jax.random import PRNGKey
from numpyro.distributions import Distribution
from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
    field_validator,
    model_validator,
)
from typing_extensions import Self

from ..typing import DeterministicParameter
from .strains import Strain


class SolverParams(BaseModel):
    """Parameters used by the ODE solver."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    ode_solver_rel_tolerance: PositiveFloat
    ode_solver_abs_tolerance: PositiveFloat


@chex.dataclass
class ODEParameters:
    """The internal representation containing parameters passed to the ODEs.

    Because ODEs work with vectors/matricies/tensors as opposed to objects,
    this internal state flattens the list of strains into the tensors of information
    separate from the `Strain` class entirely.
    """

    strain_interactions: chex.ArrayDevice
    betas: chex.ArrayDevice


class TransmissionParams(BaseModel):
    """Transmission Parameters for the respiratory model."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    strain_interactions: dict[
        str,
        dict[str, NonNegativeFloat | Distribution | DeterministicParameter],
    ]
    strains: List[Strain]

    @model_validator(mode="after")
    def _validate_strain_interactions(self) -> Self:
        strain_names = [strain.strain_name for strain in self.strains]
        for infecting_strain in strain_names:
            for recovered_from_strain in strain_names:
                assert (
                    infecting_strain in self.strain_interactions.keys()
                ), f"""{infecting_strain} not found in first level of the
                    strain_interactions_dictionary, , every strain should
                    have an interaction value against all other strains, found :
                    {list(self.strain_interactions.keys())}"""
                assert (
                    recovered_from_strain
                    in self.strain_interactions[infecting_strain]
                ), f"""unable to find {recovered_from_strain} within
                    strain_interactions[{infecting_strain}], every strain should
                    have an interaction value against all other strains."""
        return self

    @field_validator("strains", mode="after")
    @classmethod
    def _validate_strains_field_matching(
        cls, strains: list[Strain]
    ) -> list[Strain]:
        strain_intro_ages = [
            strain.introduction_ages
            for strain in strains
            if strain.is_introduced
        ]
        assert all(
            [compare == strain_intro_ages[0] for compare in strain_intro_ages]
        ), "currently DynODE requires all strains have matching introduction_ages."
        # Fields to check for consistency (excluding introduction_* fields)
        optional_fields_to_check = [
            "exposed_to_infectious",
            "vaccine_efficacy",
        ]

        # whether or not the fields exist in one or more of the Strains
        field_presence_tracker = {
            field_name: False for field_name in optional_fields_to_check
        }

        # Check each strain and update tracker
        for strain in strains:
            for field_name in optional_fields_to_check:
                if getattr(strain, field_name) is not None:
                    field_presence_tracker[field_name] = True

        # Validate that all required fields are consistently set across all strains
        for field_name in optional_fields_to_check:
            if field_presence_tracker[field_name]:
                for strain in strains:
                    if getattr(strain, field_name) is None:
                        raise AssertionError(
                            f"if {field_name} is set within one strain it must be set in all of them."
                        )

        return strains


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
