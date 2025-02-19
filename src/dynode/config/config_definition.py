# from dataclasses import dataclass
from datetime import date
from typing import Annotated, Callable, List, Optional, Union

# import numpyro.distributions as dist
from jax import Array
from jax import numpy as jnp
from numpyro.distributions import Distribution
from numpyro.infer import MCMC, SVI
from pydantic import (
    BaseModel,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
    StringConstraints,
    model_validator,
)
from typing_extensions import Self

from dynode import CompartmentGradiants

from .bins import AgeBin
from .dimension import Dimension


class Compartment(BaseModel):
    """Defines a single compartment of an ODE model"""

    name: str
    dimensions: List[Dimension]
    values: Optional[Array] = None

    @model_validator(mode="after")
    def shape_match(self) -> Self:
        """Sets default values if unspecified, asserts dimensions and values shape matches."""
        target_values_shape = tuple(*[len(d_i for d_i in self.dimensions)])
        if self.values is not None:
            assert target_values_shape == self.values.shape
        else:
            # fill with default for now, values filled in at runtime.
            self.values = jnp.zeros(target_values_shape)
        return self


class Strain(BaseModel):
    """A strain in the ODE model, optionally introduced from external population."""

    strain_name: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, to_lower=True),
    ]
    r0: Union[NonNegativeFloat, Distribution]
    infectious_period: PositiveFloat
    exposed_to_infectious: Optional[PositiveFloat]
    vaccine_efficacy: List[NonNegativeFloat]
    is_introduced: bool = False
    introduction_time: Optional[Union[date, NonNegativeFloat, Distribution]]
    introduction_percentage: Optional[Union[PositiveFloat, Distribution]]
    introduction_scale: Optional[Union[PositiveFloat, Distribution]]
    introduction_ages: Optional[List[AgeBin]]


class ParamStore(BaseModel):
    strains: List[Strain]
    strain_interactions: dict[str, dict[str, NonNegativeFloat]]
    ode_solver_rel_tolerance: PositiveFloat
    ode_solver_abs_tolerance: PositiveFloat


class Initializer(BaseModel):
    description: str
    initialize_date: date
    population_size: PositiveInt

    def get_initial_state(
        self,
        compartments: list[Compartment],
        initial_infection_scale: NonNegativeFloat,
    ) -> list[Compartment]:
        """Fill in compartments with values summing to `population_size`.

        Parameters
        ----------
        compartments : list[Compartment]
            compartments whose values to fill in.

        Returns
        -------
        list[Compartment]
            input compartments with values filled in with compartments
            at `initialize_date`.

        Raises
        ------
        NotImplementedError
            Each initializer must implement their own `get_initial_state()`
            based on the available data streams on the `initialize_date`

        """
        raise NotImplementedError(
            "implement functionality to get initial state"
        )


class CompartmentalModel(BaseModel):
    initializer: Initializer
    compartments: List[Compartment]
    parameters: ParamStore
    # passed to diffrax.diffeqsolve
    ode_function: Callable[
        [List[Compartment], PositiveFloat, ParamStore], CompartmentGradiants
    ]
    # includes observation method, specified at runtime.
    inference_method: Optional[MCMC | SVI] = None
