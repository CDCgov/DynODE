import numpyro.distributions as dist
import numpyro.distributions.constraints as constraints
import numpyro.distributions.transforms as transforms

from .config_definition import (
    AgeBin,
    CategoricalBin,
    Compartment,
    CompartmentalModel,
    Dimension,
    Strain,
)


class SEIPModel(CompartmentalModel):
    def __init__(self):
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
        age_dimension = Dimension(
            name="age",
            bins=[
                AgeBin(min_value=0, max_value=17),
                AgeBin(min_value=18, max_value=49),
                AgeBin(min_value=50, max_value=64),
                AgeBin(min_value=65, max_value=99),
            ],
        )
        immune_history_dimension = Dimension(
            name="hist",
            bins=[
                CategoricalBin(name=strain.strain_name) for strain in strains
            ],
        )
        immune_history_full = Dimension(
            name="hist", bins=immune_history_dimension
        )
        s_compartment = Compartment(
            name="s",
            dimensions=[age_dimension, immune_history_full],
        )
        self.compartments = [s_compartment]
