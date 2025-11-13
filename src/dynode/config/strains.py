"""Strain types for ODE compartment models."""

from datetime import date
from typing import Any, Dict, List, Optional, Union

import numpyro
from jax.typing import ArrayLike
from numpyro.distributions import Distribution
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    PositiveFloat,
    field_validator,
)

from dynode.typing import DynodeName

from .bins import AgeBin
from .deterministic_parameter import DeterministicParameter


class Strain(BaseModel):
    """A strain in the ODE model, optionally introduced from external population."""

    # allow Distribution objects within Strains
    model_config = ConfigDict(arbitrary_types_allowed=True)

    strain_name: DynodeName = Field(
        description="Strain name, no leading numbers or special characters."
    )
    r0: Union[
        NonNegativeFloat, ArrayLike, Distribution, DeterministicParameter
    ] = Field(
        description="""Strain reproduction number used to calculate transmission rate."""
    )
    infectious_period: Union[PositiveFloat, ArrayLike, Distribution] = Field(
        description="""Average number of days a freshly infectious population
        stays infectious for."""
    )
    exposed_to_infectious: Optional[PositiveFloat] = Field(
        default=None,
        description="""Average number of days between exposure to this strain
          before a population gains the ability to transmit it to others""",
    )
    vaccine_efficacy: Optional[dict[int, NonNegativeFloat]] = Field(
        default=None,
        description="""Dictionary mapping integer number of tracked vaccine
        doses to the protection against infection from this strain before
        immune waning is calculated.
        0 = No protection afforded by this dose
        1.0 = This dose count grants full immunity before waning.""",
    )
    is_introduced: bool = Field(
        default=False,
        description="""Whether or not this strain is introduced to the tracked
        population via some untracked interactions during the simulation.
        New strains are introduced to the tracked population this way.""",
    )
    introduction_time: Optional[
        Union[
            date,
            NonNegativeFloat,
            ArrayLike,
            Distribution,
            DeterministicParameter,
        ]
    ] = Field(
        default=None,
        description="""Date of peak external infectious population mixing.
        External infectious individuals are slowly mixed in with the
        tracked population to avoid discontinuities in the ode solver.
        Only utilized if `is_introduced` is True. """,
    )
    introduction_percentage: Optional[
        Union[PositiveFloat, ArrayLike, Distribution, DeterministicParameter]
    ] = Field(
        default=None,
        description="""Size of the untracked external population relative
            to tracked model population.
            0.0 = No External introductions.
            0.05 = external population size is 5 percent of tracked population size.
            1.0 = external population has equal size to model population.
            Only utilized if `is_introduced` is True. """,
    )
    introduction_scale: Optional[
        Union[PositiveFloat, ArrayLike, Distribution, DeterministicParameter]
    ] = Field(
        default=None,
        description="""How rapidly external infectious population mixes with
        tracked population. Measured as a standard deviation of a normal distribution.
        4.0 = 67 percent of new strain introductions occur within 8 day window around `introduction_time`

        Only utilized if `is_introduced` is True.""",
    )
    introduction_ages: Optional[List[AgeBin]] = Field(
        default=None,
        description="""Age structure of the external infectious population.
        This is important as external populations still mix with tracked persons
        according to the chosen contact matrix.
        Only utilized if `is_introduced` is True.""",
    )

    introduction_ages_mask_vector: Optional[List[int]] = Field(
        default=None,
        description="""PRIVATE: A field automatically set when this strain enters a
    CompartmentalModel, after validating that `introduction_ages`
    contains valid bins, this field encodes which age bins
    will be introduced, None= All zeros for each age bin.""",
    )

    interactions: Dict[
        str, Union[float, ArrayLike, Distribution, DeterministicParameter]
    ] = Field(
        default_factory=dict,
        description="Optional per-pair interaction overrides for this source strain.",
    )

    @field_validator("interactions", mode="after")
    def _no_empty_keys(cls, v: Dict[str, object]) -> Dict[str, object]:
        if "" in v:
            raise ValueError(
                "Interactions cannot have empty target strain names."
            )
        return v

    def sample_distributions(self):
        obj_dict = self.model_dump()

        for key, value in obj_dict.items():
            if issubclass(type(value), Distribution):
                numpyro.sample(value)

    # Need to support more than one parameter set for this function
    # the dependent parameters could be in multiple other sets
    def resolve_deterministic(self, dependent_parameter_set: dict[str, Any]):
        obj_dict = self.model_dump()

        for key, value in obj_dict.items():
            if isinstance(value, DeterministicParameter):
                numpyro.determinisitc(value.resolve(dependent_parameter_set))
