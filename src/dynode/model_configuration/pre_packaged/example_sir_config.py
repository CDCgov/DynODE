"""A basic static SIR model configuration for demonstration purposes."""

from datetime import date

import jax.numpy as jnp
import numpyro.distributions as dist

from dynode.model_configuration import (
    Bin,
    Compartment,
    Dimension,
    Initializer,
    Params,
    SimulationConfig,
    SolverParams,
    Strain,
    TransmissionParams,
)
from dynode.typing import CompartmentState


class SIRInitializer(Initializer):
    """Initializer for SIR model, setting initial conditions for compartments."""

    def __init__(self):
        super().__init__(
            description="An SIR initalizer",
            initialize_date=date(2025, 3, 13),
            population_size=1000,
        )

    def get_initial_state(self, **kwargs) -> CompartmentState:
        _: SimulationConfig = kwargs["SIRConfig"]
        # SimulationConfig has no impact on initial state in this example
        return (
            self.population_size * jnp.array([0.99]),
            self.population_size * jnp.array([0.01]),
            jnp.array([0.00]),
        )


# TODO add young/old age structure.
# TODO add trivial contact matrix.
class SIRConfig(SimulationConfig):
    def __init__(self):
        """Set parameters for a static SIR compartmental model.

        This includes compartment shape, initializer, and solver/transmission parameters."""
        dimension = Dimension(name="value", bins=[Bin(name="value")])
        s = Compartment(name="s", dimensions=[dimension])
        i = Compartment(name="i", dimensions=[dimension])
        r = Compartment(name="r", dimensions=[dimension])
        strain = [
            Strain(strain_name="example_strain", r0=2, infectious_period=7.0)
        ]
        parameters = Params(
            solver_params=SolverParams(),
            transmission_params=TransmissionParams(
                strains=strain,
                strain_interactions={
                    "example_strain": {"example_strain": 1.0}
                },
            ),
        )
        super().__init__(
            compartments=[s, i, r],
            initializer=SIRInitializer(),
            parameters=parameters,
        )


class SIRInferedConfig(SIRConfig):
    def __init__(self):
        """Set parameters for a infered SIR compartmental model.

        This includes compartment shape, initializer, and solver/transmission parameters."""
        # build the static version then replace the strain with
        # one modeled by some proposed priors instead.
        super().__init__()

        self.parameters.transmission_params.strains = [
            Strain(
                strain_name="example_strain",
                r0=dist.TransformedDistribution(
                    dist.Beta(0.5, 0.5),
                    dist.transforms.AffineTransform(1.5, 1),
                ),
                infectious_period=dist.TruncatedNormal(
                    loc=8, scale=2, low=2, high=15
                ),
            )
        ]
