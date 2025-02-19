from typing import List, Union

from pydantic import BaseModel

from .bins import AgeBin, CategoricalBin, DiscretizedPositiveIntBin
from .config_definition import Strain


class Dimension(BaseModel):
    """A dimension for a compartment"""

    name: str
    bins: Union[
        List[CategoricalBin],
        List[DiscretizedPositiveIntBin],
        List[AgeBin],
    ]


class VaccinationDimension(Dimension):
    """A vaccination dimension of a compartment, supporting ordinal (and optionally seasonal) vaccinations."""

    def __init__(
        self, max_ordinal_vaccinations: int, seasonal_vaccination: bool = False
    ):
        self.name = "vax"
        if seasonal_vaccination:
            max_ordinal_vaccinations += 1
        self.bins = [
            DiscretizedPositiveIntBin(min_value=vax_count, max_value=vax_count)
            for vax_count in range(max_ordinal_vaccinations + 1)
        ]


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
