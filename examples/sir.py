from datetime import date

import chex
import jax.numpy as jnp
import matplotlib.pyplot as plt

from dynode.config import (
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
from dynode.simulation import AbstractODEParams, simulate
from dynode.typing import CompartmentState


# --- Minimal Initializer ---
class SimpleSIRInitializer(Initializer):
    def __init__(self):
        super().__init__(
            description="Simple SIR initializer",
            initialize_date=date(2022, 2, 11),
            population_size=1.0,
        )

    def get_initial_state(
        self, s_0=0.9, i_0=0.1, r_0=0.0, **kwargs
    ) -> CompartmentState:
        s_0 = jnp.array([s_0])
        i_0 = jnp.array([i_0])
        r_0 = jnp.array([r_0])
        return (s_0, i_0, r_0)


# --- Minimal Config ---
def get_config(r_0=2.0, infectious_period=7.0) -> SimulationConfig:
    dimension = Dimension(name="age", bins=[Bin(name="all")])
    s = Compartment(name="s", dimensions=[dimension])
    i = Compartment(name="i", dimensions=[dimension])
    r = Compartment(name="r", dimensions=[dimension])

    strain = [
        Strain(strain_name="test", r0=r_0, infectious_period=infectious_period)
    ]
    contact_matrix = jnp.array([[1.0]])
    parameters = Params(
        solver_params=SolverParams(),
        transmission_params=TransmissionParams(
            strains=strain,
            strain_interactions={"test": {"test": 1.0}},
            contact_matrix=contact_matrix,
        ),
    )

    config = SimulationConfig(
        compartments=[s, i, r],
        initializer=SimpleSIRInitializer(),
        parameters=parameters,
    )
    return config


# --- ODE Params ---
@chex.dataclass
class SIR_ODEParams(AbstractODEParams):
    beta: chex.ArrayDevice
    gamma: chex.ArrayDevice


# --- SIR ODE ---
def sir_ode(t: float, state: CompartmentState, p: SIR_ODEParams):
    s, i, r = state
    N = s + i + r
    ds = -p.beta * s * i / N
    di = p.beta * s * i / N - p.gamma * i
    dr = p.gamma * i
    return (ds, di, dr)


def get_odeparams(config: SimulationConfig) -> SIR_ODEParams:
    """Transform and vectorize transmission parameters into ODE parameters."""
    strain = config.parameters.transmission_params.strains[0]
    beta = strain.r0 / strain.infectious_period
    gamma = 1.0 / strain.infectious_period
    return SIR_ODEParams(beta=beta, gamma=gamma)


# --- Run and Plot ---
if __name__ == "__main__":
    config = get_config()

    sol = simulate(
        ode=sir_ode,
        duration_days=150,
        initial_state=config.initializer.get_initial_state(),
        ode_parameters=get_odeparams(config),
        solver_parameters=config.parameters.solver_params,
    )
    # sol.ys is a tuple of arrays (s, i, r), each shape (timesteps, 1)
    s, i, r = [arr.squeeze() for arr in sol.ys]
    t = sol.ts
    plt.plot(t, s, label="Susceptible")
    plt.plot(t, i, label="Infectious")
    plt.plot(t, r, label="Recovered")
    plt.xlabel("Days")
    plt.ylabel("Population")
    plt.legend()
    plt.title("Simple SIR Model")
    plt.show()
