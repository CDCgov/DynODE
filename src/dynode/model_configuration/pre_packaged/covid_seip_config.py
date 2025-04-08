"""Config for fitting covid SEIP models from feb 11th 2022 onwards."""

import math
from datetime import date

import numpyro.distributions as dist
import numpyro.distributions.constraints as constraints
import numpyro.distributions.transforms as transforms

from ...typing import DeterministicParameter, SimulationDate
from ..bins import AgeBin, Bin
from ..config_definition import (
    Compartment,
    Initializer,
    Params,
    SimulationConfig,
)
from ..dimension import (
    Dimension,
    LastStrainImmuneHistoryDimension,
    VaccinationDimension,
    WaneDimension,
)
from ..params import SolverParams, TransmissionParams
from ..strains import Strain


class SEIPCovidConfig(SimulationConfig):
    """Covid model with partial susceptibility."""

    def __init__(self):
        """Initialize the SEIP covid config."""
        strains = self._create_strains()
        compartments = self._create_compartments(strains)
        param_store = self._create_param_store(strains)
        # Here you pass in a subclass of the initializer() class with you particular behavior
        initializer = Initializer(
            description="initializer for feb 11 2022",
            initialize_date=date(2022, 2, 11),
            population_size=100000,
        )
        super().__init__(
            initializer=initializer,
            compartments=compartments,
            parameters=param_store,
        )

    @property
    def s(self) -> Compartment:
        """The Susceptible compartment of the model."""
        return self.compartments[0]

    @property
    def e(self) -> Compartment:
        """The Exposed compartment of the model."""
        return self.compartments[1]

    @property
    def i(self) -> Compartment:
        """The Infectious compartment of the model."""
        return self.compartments[2]

    @property
    def c(self) -> Compartment:
        """The Cumulative compartment of the model."""
        return self.compartments[3]

    def _create_param_store(self, strains: list[Strain]) -> Params:
        # ignore mypy when additional parameters not found in TransmissionParams
        transmission_params = TransmissionParams(  # type:ignore
            strain_interactions_2_steps=dist.TransformedDistribution(
                base_distribution=dist.Beta(60, 240),
                transforms=transforms.AffineTransform(
                    loc=0.5, scale=0.5, domain=constraints.unit_interval
                ),
            ),
            strain_interactions_3_steps=dist.Beta(75, 225),
            strains=strains,
            strain_interactions={
                "omicron": {
                    "omicron": 0.75,
                    "ba2ba5": 1.0,
                    "xbb": 1.0,
                    "jn1": 1.0,
                },
                "ba2ba5": {
                    "omicron": dist.TransformedDistribution(
                        base_distribution=dist.Beta(60, 240),
                        transforms=transforms.AffineTransform(
                            loc=0.5,
                            scale=0.5,
                            domain=constraints.unit_interval,
                        ),
                    ),
                    "ba2ba5": 1.0,
                    "xbb": 1.0,
                    "jn1": 1.0,
                },
                "xbb": {
                    "omicron": DeterministicParameter(
                        depends_on="strain_interactions_2_steps"
                    ),
                    "ba2ba5": dist.TransformedDistribution(
                        base_distribution=dist.Beta(120, 180),
                        transforms=transforms.AffineTransform(
                            loc=0.5,
                            scale=0.5,
                            domain=constraints.unit_interval,
                        ),
                    ),
                    "xbb": 1.0,
                    "jn1": 1.0,
                },
                "jn1": {
                    "omicron": DeterministicParameter(
                        depends_on="strain_interactions_3_steps"
                    ),
                    "ba2ba5": DeterministicParameter(
                        depends_on="strain_interactions_2_steps"
                    ),
                    "xbb": dist.TransformedDistribution(
                        base_distribution=dist.Beta(120, 180),
                        transforms=transforms.AffineTransform(
                            loc=0.5,
                            scale=0.5,
                            domain=constraints.unit_interval,
                        ),
                    ),
                    "jn1": 1.0,
                },
            },
        )
        return Params(
            transmission_params=transmission_params,
            solver_params=SolverParams(),
        )

    def _create_strains(self) -> list[Strain]:
        strains = [
            Strain(
                strain_name="omicron",
                r0=dist.TransformedDistribution(
                    base_distribution=dist.Beta(
                        concentration1=140, concentration0=60
                    ),
                    transforms=transforms.AffineTransform(
                        loc=1.0, scale=2.0, domain=constraints.unit_interval
                    ),
                ),
                infectious_period=7.0,
                exposed_to_infectious=3.6,
                vaccine_efficacy={0: 0, 1: 0.35, 2: 0.70},
            ),
            Strain(
                strain_name="ba2ba5",
                r0=dist.TransformedDistribution(
                    base_distribution=dist.Beta(35, 65),
                    transforms=transforms.AffineTransform(
                        loc=2.0, scale=2.0, domain=constraints.unit_interval
                    ),
                ),
                infectious_period=7.0,
                exposed_to_infectious=3.6,
                vaccine_efficacy={0: 0, 1: 0.30, 2: 0.60},
                is_introduced=True,
                introduction_time=dist.TruncatedNormal(
                    loc=SimulationDate(2022, 3, 3),
                    scale=5,
                    low=SimulationDate(2022, 2, 21),
                ),
                introduction_scale=15,
                introduction_percentage=0.02,
                introduction_ages=[AgeBin(min_value=18, max_value=49)],
            ),
            Strain(
                strain_name="xbb",
                r0=dist.TransformedDistribution(
                    base_distribution=dist.Beta(30, 70),
                    transforms=transforms.AffineTransform(
                        loc=2.0, scale=3.0, domain=constraints.unit_interval
                    ),
                ),
                infectious_period=7.0,
                exposed_to_infectious=3.6,
                vaccine_efficacy={0: 0, 1: 0.30, 2: 0.60},
                is_introduced=True,
                introduction_time=dist.TruncatedNormal(
                    loc=SimulationDate(2022, 9, 29),
                    scale=5,
                    low=SimulationDate(2022, 8, 20),
                ),
                introduction_scale=15,
                introduction_percentage=0.02,
                introduction_ages=[AgeBin(min_value=18, max_value=49)],
            ),
            Strain(
                strain_name="jn1",
                r0=dist.TransformedDistribution(
                    base_distribution=dist.Beta(30, 70),
                    transforms=transforms.AffineTransform(
                        loc=2.0, scale=3.0, domain=constraints.unit_interval
                    ),
                ),
                infectious_period=7.0,
                exposed_to_infectious=3.6,
                vaccine_efficacy={0: 0, 1: 0.095, 2: 0.19},
                is_introduced=True,
                introduction_time=dist.TruncatedNormal(
                    loc=SimulationDate(2023, 11, 13),
                    scale=5,
                    low=SimulationDate(2023, 10, 4),
                ),
                introduction_scale=15,
                introduction_percentage=0.02,
                introduction_ages=[AgeBin(min_value=18, max_value=49)],
            ),
        ]
        return strains

    def _create_compartments(self, strains: list[Strain]) -> list[Compartment]:
        age_dimension = Dimension(
            name="age",
            bins=[
                AgeBin(min_value=0, max_value=17),
                AgeBin(min_value=18, max_value=49),
                AgeBin(min_value=50, max_value=64),
                AgeBin(min_value=65, max_value=99),
            ],
        )
        immune_history_dimension = LastStrainImmuneHistoryDimension(
            strains=strains
        )
        vaccination_dimension = VaccinationDimension(
            max_ordinal_vaccinations=2, seasonal_vaccination=False
        )
        waning_dimension = WaneDimension(
            waiting_times=[70, 70, 70, math.inf],
            base_protections=[1.0, 1.0, 1.0, 0.0],
        )
        infecting_strain_dimension = Dimension(
            name="strain",
            bins=[Bin(name=strain.strain_name) for strain in strains],
        )
        s_compartment = Compartment(
            name="s",
            dimensions=[
                age_dimension,
                immune_history_dimension,
                vaccination_dimension,
                waning_dimension,
            ],
        )
        e_compartment = Compartment(
            name="e",
            dimensions=[
                age_dimension,
                immune_history_dimension,
                vaccination_dimension,
                infecting_strain_dimension,
            ],
        )
        i_compartment = Compartment(
            name="i",
            dimensions=[
                age_dimension,
                immune_history_dimension,
                vaccination_dimension,
                infecting_strain_dimension,
            ],
        )
        c_compartment = Compartment(
            name="c",
            dimensions=[
                age_dimension,
                immune_history_dimension,
                vaccination_dimension,
                waning_dimension,
                infecting_strain_dimension,
            ],
        )
        return [
            s_compartment,
            e_compartment,
            i_compartment,
            c_compartment,
        ]
