"""Config for fitting covid SEIP models from feb 11th 2022 onwards."""

from datetime import date

import numpyro.distributions as dist
import numpyro.distributions.constraints as constraints
import numpyro.distributions.transforms as transforms

from dynode.model_odes.seip_model import seip_ode

from ...typing import DeterministicParameter
from ..bins import AgeBin, Bin, WaneBin
from ..config_definition import (
    Compartment,
    CompartmentalConfig,
    Initializer,
    Params,
)
from ..dimension import (
    Dimension,
    LastStrainImmuneHistory,
    VaccinationDimension,
)
from ..params import SolverParams, TransmissionParams
from ..strains import Strain


class SEIPCovidConfig(CompartmentalConfig):
    """Covid model with partial susceptibility."""

    def __init__(self):
        """Initialize the SEIP covid config."""
        strains = self._get_strains()
        compartments = self._get_compartments(strains)
        param_store = self._get_param_store(strains)
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
            ode_function=seip_ode,
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

    def _get_param_store(self, strains: list[Strain]) -> Params:
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
            solver_params=SolverParams(
                ode_solver_rel_tolerance=1e-5,
                ode_solver_abs_tolerance=1e-6,
            ),
        )

    def _get_strains(self) -> list[Strain]:
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
                    loc=20, scale=5, low=10
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
                    loc=230, scale=5, low=190
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
                    loc=640, scale=5, low=600
                ),
                introduction_scale=15,
                introduction_percentage=0.02,
                introduction_ages=[AgeBin(min_value=18, max_value=49)],
            ),
        ]
        return strains

    def _get_compartments(self, strains: list[Strain]) -> list[Compartment]:
        age_dimension = Dimension(
            name="age",
            bins=[
                AgeBin(min_value=0, max_value=17),
                AgeBin(min_value=18, max_value=49),
                AgeBin(min_value=50, max_value=64),
                AgeBin(min_value=65, max_value=99),
            ],
        )
        immune_history_dimension = LastStrainImmuneHistory(strains=strains)
        vaccination_dimension = VaccinationDimension(
            max_ordinal_vaccinations=2, seasonal_vaccination=False
        )
        waning_dimension = Dimension(
            name="wane",
            bins=[
                WaneBin(name="w0", waning_time=70, waning_protection=1.0),
                WaneBin(name="w1", waning_time=70, waning_protection=1.0),
                WaneBin(name="w2", waning_time=70, waning_protection=1.0),
                WaneBin(name="w3", waning_time=0, waning_protection=0.0),
            ],
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
