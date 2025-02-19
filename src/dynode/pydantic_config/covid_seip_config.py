"""Config for fitting covid SEIP models from feb 11th 2022 onwards."""

from datetime import date

import numpyro.distributions as dist
import numpyro.distributions.constraints as constraints
import numpyro.distributions.transforms as transforms

from dynode.model_odes.seip_model import seip_ode

from .bins import AgeBin, CategoricalBin, WaneBin
from .config_definition import (
    Compartment,
    CompartmentalModel,
    Initializer,
    ParamStore,
    Strain,
)
from .dimension import Dimension, LastStrainImmuneHistory, VaccinationDimension


class SEIPCovidModel(CompartmentalModel):
    """Covid model with partial susceptibility."""

    def __init__(self):
        """Initialize the SEIP covid config."""
        strains = self._get_strains()
        compartments = self._get_compartments(strains)
        param_store = self._get_param_store()
        # Here you pass in a subclass of the initializer() class with you particular behavior
        self.initializer = Initializer(
            description="initializer for feb 11 2022",
            initialize_date=date(2022, 2, 11),
            population_size=100000,
        )
        self.compartments = compartments
        self.parameters = param_store
        # here you pass in your ODEs which take in the same compartment shapes you specified above
        self.ode_function = seip_ode

    def _get_param_store(self, strains) -> ParamStore:
        return ParamStore(
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
                    "omicron": 0.22,  # todo, figure out how to specify linkage between this and [jn1][ba2ba5]
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
                    "omicron": 0.33,
                    "ba2ba5": 0.22,
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
            ode_solver_rel_tolerance=1e-5,
            ode_solver_abs_tolerance=1e-6,
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
                vaccine_efficacy=[0, 0.35, 0.70],
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
                vaccine_efficacy=[0, 0.30, 0.60],
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
                vaccine_efficacy=[0, 0.30, 0.60],
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
                vaccine_efficacy=[0, 0.095, 0.19],
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

    def _get_compartments(self, strains) -> list[Compartment]:
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
            max_ordinal_vaccinations=2, seasonal_vaccination=True
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
            bins=[
                CategoricalBin(name=strain.strain_name) for strain in strains
            ],
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
