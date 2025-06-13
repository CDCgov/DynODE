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


class SimpleSEIRSInitializer(Initializer):
    def __init__(self):
        super().__init__(
            description="Simple SEIRS initializer",
            initialize_date=date(2022, 2, 11),
            population_size=1.0,
        )

    def get_initial_state(
        self, s_0=0.99, e_0=0.0, i_0=0.01, r_0=0.0, **kwargs
    ) -> CompartmentState:
        s_0 = jnp.array([s_0])
        e_0 = jnp.array([e_0])
        i_0 = jnp.array([i_0])
        r_0 = jnp.array([r_0])
        return (s_0, e_0, i_0, r_0)


def get_config(
    r_0=2.0, infectious_period=7.0, latent_period=3.0, waning_period=60.0
) -> SimulationConfig:
    dimension = Dimension(name="age", bins=[Bin(name="all")])
    s = Compartment(name="s", dimensions=[dimension])
    e = Compartment(name="e", dimensions=[dimension])
    i = Compartment(name="i", dimensions=[dimension])
    r = Compartment(name="r", dimensions=[dimension])

    strain = [
        Strain(strain_name="test", r0=r_0, infectious_period=infectious_period)
    ]
    contact_matrix = jnp.array([[1.0]])
    transmission_params = TransmissionParams(
        strains=strain,
        strain_interactions={"test": {"test": 1.0}},
        contact_matrix=contact_matrix,
        latent_period=latent_period,
        waning_period=waning_period,
    )
    # Store extra parameters in transmission_params
    # setattr(transmission_params, "latent_period", latent_period)
    # setattr(transmission_params, "waning_period", waning_period)

    parameters = Params(
        solver_params=SolverParams(),
        transmission_params=transmission_params,
    )

    config = SimulationConfig(
        compartments=[s, e, i, r],
        initializer=SimpleSEIRSInitializer(),
        parameters=parameters,
    )
    return config


@chex.dataclass
class SEIRS_ODEParams(AbstractODEParams):
    beta: chex.ArrayDevice
    gamma: chex.ArrayDevice
    sigma: chex.ArrayDevice
    omega: chex.ArrayDevice


def seirs_ode(t: float, state: CompartmentState, p: SEIRS_ODEParams):
    s, e, i, r = state
    N = s + e + i + r
    ds = -p.beta * s * i / N + p.omega * r
    de = p.beta * s * i / N - p.sigma * e
    di = p.sigma * e - p.gamma * i
    dr = p.gamma * i - p.omega * r
    return (ds, de, di, dr)


def get_seirs_odeparams(config: SimulationConfig) -> SEIRS_ODEParams:
    strain = config.parameters.transmission_params.strains[0]
    beta = strain.r0 / strain.infectious_period
    gamma = 1.0 / strain.infectious_period
    sigma = 1.0 / config.parameters.transmission_params.latent_period
    omega = 1.0 / config.parameters.transmission_params.waning_period
    return SEIRS_ODEParams(beta=beta, gamma=gamma, sigma=sigma, omega=omega)


if __name__ == "__main__":
    config = get_config()
    sol = simulate(
        ode=seirs_ode,
        duration_days=1500,
        initial_state=config.initializer.get_initial_state(),
        ode_parameters=get_seirs_odeparams(config),
        solver_parameters=config.parameters.solver_params,
    )
    s, e, i, r = [arr.squeeze() for arr in sol.ys]
    t = sol.ts
    plt.plot(t, s, label="Susceptible")
    plt.plot(t, e, label="Exposed")
    plt.plot(t, i, label="Infectious")
    plt.plot(t, r, label="Recovered")
    plt.xlabel("Days")
    plt.ylabel("Population")
    plt.legend()
    plt.title("Simple SEIRS Model")
    plt.show()
