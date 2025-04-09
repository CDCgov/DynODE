"""A basic static SIR model configuration for demonstration purposes."""

from datetime import date

import jax.numpy as jnp

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


class SIRConfig(SimulationConfig):
    def __init__(self):
        """Set parameters for an SIR compartmental model.

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
