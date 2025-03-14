# An Example of how to simulate a basic SIR compartmental model using the dynode package
# Including all the class setup
# %% imports, mostly for class creation
from datetime import date
from functools import partial

import chex
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from diffrax import Solution, is_okay
from jax import Array

from dynode.model_configuration import (
    Compartment,
    Initializer,
    SimulationConfig,
)
from dynode.model_configuration.bins import Bin
from dynode.model_configuration.dimension import Dimension
from dynode.model_configuration.params import (
    Params,
    SolverParams,
    TransmissionParams,
)
from dynode.model_configuration.strains import Strain
from dynode.odes import AbstractODEParams, ODEBase


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
            solver_params=SolverParams(
                ode_solver_rel_tolerance=1e-7,
                ode_solver_abs_tolerance=1e-8,
                # constant_step_size=0.5,
                max_steps=100000,
            ),
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


class SIR_ODE(ODEBase):
    partial(jax.jit, static_argnums=0)

    def __call__(
        self,
        compartments: tuple[Array],
        t: float,  # unused in this basic example, useful for time-varying parameters
        p: SIR_ODEParams,  # notice that we are passing SIR_ODEParams here.
    ):
        s, i, r = compartments
        ds = -p.beta * s * i
        dr = i * p.gamma
        # jax.debug.print(
        #     "t {}, s: {}, i {}, r {}, ds: {}, dr: {}", t, s, i, r, ds, dr
        # )
        return [ds, -ds - dr, dr]


# %% simulation

# set up config
config = SIRConfig()
ode_params = get_odeparams(config.parameters.transmission_params)

# set up odes
ode = SIR_ODE()
# we need just the jax arrays for the initial state to the ODEs
initial_state = [
    compartment.values
    for compartment in config.initializer.get_initial_state(SIRConfig=config)
]
# solve the odes for 100 days

solution: Solution = ode.solve(
    initial_state=initial_state,
    solver_parameters=config.parameters.solver_params,
    ode_parameters=ode_params,
    duration_days=100,
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
