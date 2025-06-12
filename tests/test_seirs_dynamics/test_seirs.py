from datetime import date

import chex
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pytest

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


# --- SEIRS Initializer ---
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


# --- SEIRS Config ---
def get_seirs_config(
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
    )
    # Store extra parameters in transmission_params
    setattr(transmission_params, "latent_period", latent_period)
    setattr(transmission_params, "waning_period", waning_period)

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


# --- ODE Params ---
@chex.dataclass
class SEIRS_ODEParams(AbstractODEParams):
    beta: chex.ArrayDevice
    gamma: chex.ArrayDevice
    sigma: chex.ArrayDevice
    omega: chex.ArrayDevice


# --- SEIRS ODE ---


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
    sigma = 1.0 / getattr(
        config.parameters.transmission_params, "latent_period", 3.0
    )
    omega = 1.0 / getattr(
        config.parameters.transmission_params, "waning_period", 60.0
    )
    return SEIRS_ODEParams(beta=beta, gamma=gamma, sigma=sigma, omega=omega)


# --- Run and Plot ---
if __name__ == "__main__":
    config = get_seirs_config()
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


def compute_seirs_endemic_equilibrium(beta, gamma, sigma, omega):
    """
    Computes the endemic equilibrium for the SEIRS model.
    Returns (S*, E*, I*, R*) for a normalized population (N=1).
    """
    S_star = gamma / beta
    denom = 1 + (gamma / sigma) + (gamma / omega)
    I_star = (1 - S_star) / denom
    E_star = (gamma / sigma) * I_star
    R_star = (gamma / omega) * I_star
    return S_star, E_star, I_star, R_star


@pytest.mark.parametrize(
    "r_0, infectious_period, latent_period, waning_period",
    [
        (2.0, 7.0, 3.0, 60.0),
        (3.0, 5.0, 2.0, 100.0),
    ],
)
def test_seirs_endemic_equilibrium(
    r_0, infectious_period, latent_period, waning_period
):
    config = get_seirs_config(
        r_0=r_0,
        infectious_period=infectious_period,
        latent_period=latent_period,
        waning_period=waning_period,
    )
    ode_params = get_seirs_odeparams(config)
    sol = simulate(
        ode=seirs_ode,
        duration_days=6000,
        initial_state=config.initializer.get_initial_state(),
        ode_parameters=ode_params,
        solver_parameters=config.parameters.solver_params,
    )
    s, e, i, r = [arr.squeeze() for arr in sol.ys]
    # Take the last 100 days and average to estimate equilibrium
    # Ensure the last 100 days are stable (not fluctuating)
    assert jnp.std(s[-100:]) < 1e-4
    assert jnp.std(e[-100:]) < 1e-4
    assert jnp.std(i[-100:]) < 1e-4
    assert jnp.std(r[-100:]) < 1e-4
    s_eq = float(s[-1])
    e_eq = float(e[-1])
    i_eq = float(i[-1])
    r_eq = float(r[-1])
    # Compute theoretical equilibrium
    S_star, E_star, I_star, R_star = compute_seirs_endemic_equilibrium(
        ode_params.beta, ode_params.gamma, ode_params.sigma, ode_params.omega
    )
    # Assert close to theoretical values
    assert pytest.approx(s_eq, rel=1e-2) == S_star
    assert pytest.approx(e_eq, rel=1e-2) == E_star
    assert pytest.approx(i_eq, rel=1e-2) == I_star
    assert pytest.approx(r_eq, rel=1e-2) == R_star
