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
    NonNegativeInt,
    PositiveFloat,
    StringConstraints,
    model_validator,
)
from typing_extensions import Self

from dynode import CompartmentGradiants


class CategoricalBin(BaseModel):
    "bin with a distinct name"

    name: str


class DiscretizedPositiveIntBin(BaseModel):
    "bin with a distinct discretized positive int mins (e.g., age)"

    min_value: NonNegativeInt
    max_value: NonNegativeInt

    @model_validator(mode="after")
    def bin_valid_side(self) -> Self:
        """Asserts that min_value <= max_value"""
        assert self.min_value <= self.max_value
        return self


class AgeBin(DiscretizedPositiveIntBin):
    pass


class WaneBin(CategoricalBin):
    waning_time: NonNegativeInt
    waning_protection: PositiveFloat


class Dimension(BaseModel):
    """A dimension for a compartment"""

    name: str
    bins: Union[
        List[CategoricalBin],
        List[DiscretizedPositiveIntBin],
        List[AgeBin],
    ]


class Compartment(BaseModel):
    """Specify a single compartment"""

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


class FullStratifiedImmuneHistory(Dimension):
    """A type of immune history which represents all possible unique infections."""

    def __init__(self, strains: list[Strain]) -> None:
        """Create a fully stratified immune history dimension."""
        self.name = "hist"
        strain_names = [s.strain_name for s in strains]
        num_strains = len(strain_names)
        all_immune_histories = []
        for i in range(2**num_strains):
            immune_hist = []
            for j in range(num_strains):
                if (i & (1 << j)) > 0:
                    immune_hist.append(strains[j])
            all_immune_histories.append("-".join(immune_hist))

        self.bins = [
            CategoricalBin(name=state) for state in all_immune_histories
        ]


class LastStrainImmuneHistory(Dimension):
    def __init__(self, strains: list[Strain]) -> None:
        """Create an immune history dimension that only tracks last infected strain."""
        self.name = "hist"
        strain_names = [s.strain_name for s in strains]
        self.bins = [CategoricalBin(name=state) for state in strain_names]


class ParamStore(BaseModel):
    strains: List[Strain]
    strain_interactions: dict[str, dict[str, NonNegativeFloat]]


class CompartmentalModel(BaseModel):
    compartments: List[Compartment]
    # passed to diffrax.diffeqsolve
    ode_function: Callable[
        [List[Compartment], PositiveFloat, ParamStore], CompartmentGradiants
    ]
    # includes observation method, specified at runtime.
    inference_method: Optional[MCMC | SVI] = None
