# An Example of how to simulate a basic SIR compartmental model using the dynode package
# Including all the class setup
# %% imports, mostly for class creation
from datetime import date

import chex
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from diffrax import Solution, is_okay

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
from dynode.odes import AbstractODEParams, simulate
from dynode.typing import CompartmentGradients, CompartmentState


# %% class definitions
class SIRInitializer(Initializer):
    """Initializer for SIR model, setting initial conditions for compartments."""

    def __init__(self):
        super().__init__(
            description="An SIR initalizer",
            initialize_date=date(2025, 3, 13),
            population_size=1,
        )

    def get_initial_state(self, **kwargs):
        config: SimulationConfig = kwargs["SIRConfig"]
        s = config.get_compartment("s")
        i = config.get_compartment("i")
        r = config.get_compartment("r")
        s.values = jnp.array([0.99])  # need jnp.array for ode solver
        i.values = jnp.array([0.01])
        r.values = jnp.array([0.00])
        return [s, i, r]


class SIRConfig(SimulationConfig):
    def __init__(self):
        """Set parameters for an SIR compartmental model.

        This includes compartment shape, initializer, and solver/transmission parameters."""
        dimension = Dimension(name="value", bins=[Bin(name="value")])
        s = Compartment(name="s", dimensions=[dimension])
        i = Compartment(name="i", dimensions=[dimension])
        r = Compartment(name="r", dimensions=[dimension])
        strain = [
            Strain(strain_name="example_strain", r0=2.0, infectious_period=7.0)
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


@chex.dataclass
class SIR_ODEParams(AbstractODEParams):
    beta: chex.ArrayDevice  # r0/infectious period
    gamma: chex.ArrayDevice  # 1/infectious period
    pass


def get_odeparams(transmission_params: TransmissionParams) -> SIR_ODEParams:
    """Transform and vectorize transmission parameters into ODE parameters."""
    strain = transmission_params.strains[0]
    beta = strain.r0 / strain.infectious_period  # infection rate
    gamma = 1 / strain.infectious_period  # recovery rate
    return SIR_ODEParams(beta=beta, gamma=gamma)


@jax.jit
def sir_ode(
    t: float, state: CompartmentState, p: SIR_ODEParams
) -> CompartmentGradients:
    s, i, _ = state
    s_to_i = p.beta * s * i
    i_to_r = i * p.gamma
    ds = -s_to_i
    di = s_to_i - i_to_r
    dr = i_to_r
    return [ds, di, dr]


# %% simulation

# set up config
config = SIRConfig()
ode_params = get_odeparams(config.parameters.transmission_params)

# we need just the jax arrays for the initial state to the ODEs
initial_state = [
    compartment.values
    for compartment in config.initializer.get_initial_state(SIRConfig=config)
]
# solve the odes for 100 days

solution: Solution = simulate(
    ode=sir_ode,
    duration_days=100,
    initial_state=initial_state,
    ode_parameters=ode_params,
    solver_parameters=config.parameters.solver_params,
)
if is_okay(solution.result):
    print("solution is okay")
    print(solution.ts)
else:
    print("solution is not okay")
    print(solution.result)
    print(solution.ts)
    print(initial_state)
    print(ode_params)
    raise Exception(str(solution.result))


# %% plot
plt.plot(solution.ys[0], label="s")
plt.plot(solution.ys[1], label="i")
plt.plot(solution.ys[2], label="r")
plt.legend()
plt.show()

# %%
