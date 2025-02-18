import numpyro.distributions as dist
import numpyro.distributions.constraints as constraints
import numpyro.distributions.transforms as transforms

from dynode.model_odes.seip_model import seip_ode

from .config_definition import (
    AgeBin,
    CategoricalBin,
    Compartment,
    CompartmentalModel,
    Dimension,
    LastStrainImmuneHistory,
    Strain,
    VaccinationDimension,
    WaneBin,
)


class SEIPModel(CompartmentalModel):
    def __init__(self):
        strains = self._get_strains()
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
                WaneBin(waning_time=70, waning_protection=1.0),
                WaneBin(waning_time=70, waning_protection=1.0),
                WaneBin(waning_time=70, waning_protection=1.0),
                WaneBin(waning_time=0, waning_protection=0.0),
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
        self.compartments = [
            s_compartment,
            e_compartment,
            i_compartment,
            c_compartment,
        ]
        self.ode_function = seip_ode

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
