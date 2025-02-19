"""Top level classes for DynODE configs."""

from datetime import date
from typing import Callable, List, Optional

from jax import Array
from jax import numpy as jnp
from numpyro.distributions import Distribution
from numpyro.infer import MCMC, SVI
from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
    model_validator,
)
from typing_extensions import Self

from dynode import CompartmentGradiants

from .dimension import Dimension
from .strains import Strain


class Compartment(BaseModel):
    """Defines a single compartment of an ODE model."""

    # allow jax array objects within Compartments
    model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True)
    name: str
    dimensions: List[Dimension]
    values: Optional[Array] = None

    @model_validator(mode="after")
    def shape_match(self) -> Self:
        """Set default values if unspecified, asserts dimensions and values shape matches."""
        target_values_shape: tuple[int, ...] = tuple(
            [len(d_i) for d_i in self.dimensions]
        )
        if self.values is not None:
            assert target_values_shape == self.values.shape
        else:
            # fill with default for now, values filled in at runtime.
            self.values = jnp.zeros(target_values_shape)
        return self

    @property
    def shape(self) -> tuple[int, ...]:
        """Get shape of the compartment."""
        return tuple([len(d_i) for d_i in self.dimensions])


class ParamStore(BaseModel):
    """Miscellaneous parameters of an ODE model."""

    # allow users to pass custom types to ParamStore
    model_config = ConfigDict(arbitrary_types_allowed=True)
    strains: List[Strain]
    strain_interactions: dict[str, dict[str, NonNegativeFloat | Distribution]]
    ode_solver_rel_tolerance: PositiveFloat
    ode_solver_abs_tolerance: PositiveFloat


class Initializer(BaseModel):
    """Initalize compartment state of an ODE model."""

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
    """An ODE compartment model configuration file."""

    # allow users to pass custom types into CompartmentalModel
    model_config = ConfigDict(arbitrary_types_allowed=True)
    initializer: Initializer
    compartments: List[Compartment]
    parameters: ParamStore
    # passed to diffrax.diffeqsolve
    ode_function: Callable[
        [List[Compartment], PositiveFloat, ParamStore], CompartmentGradiants
    ]
    # includes observation method, specified at runtime.
    inference_method: Optional[MCMC | SVI] = None
