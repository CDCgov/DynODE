# from dataclasses import dataclass
from datetime import date
from typing import Annotated, List, Optional, Union

from numpyro.distributions import Distribution
from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeFloat,
    PositiveFloat,
    StringConstraints,
)

from .bins import AgeBin


class Strain(BaseModel):
    """A strain in the ODE model, optionally introduced from external population."""

    # allow Distribution objects within Strains
    model_config = ConfigDict(arbitrary_types_allowed=True)

    strain_name: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, to_lower=True),
    ]
    r0: Union[NonNegativeFloat, Distribution]
    infectious_period: PositiveFloat
    exposed_to_infectious: Optional[PositiveFloat]
    vaccine_efficacy: List[NonNegativeFloat]
    is_introduced: bool = False
    introduction_time: Optional[
        Union[date, NonNegativeFloat, Distribution]
    ] = None
    introduction_percentage: Optional[Union[PositiveFloat, Distribution]] = (
        None
    )
    introduction_scale: Optional[Union[PositiveFloat, Distribution]] = None
    introduction_ages: Optional[List[AgeBin]] = None
