"""An example of SEIRS model with seasonal forcing using Dynode.

This example extends the SEIRS model to include seasonal forcing
by modifying the transmission rate based on a sinusoidal function.

To achieve this, we define a `SeasonalityParams` dataclass as a subclass of our
ODE parameters, then pass these parameters to the ODE function which calls
upon the seasonal function to adjust the transmission rate."""

import chex
import jax.numpy as jnp
import matplotlib.pyplot as plt

from dynode import SimulationConfig, Strain
from dynode.simulation import AbstractODEParams, simulate
from dynode.typing import CompartmentState

from .seirs import get_config  # Import your existing config function


# --- Seasonality Params ---
@chex.dataclass
class SeasonalityParams:
    forcing_amp: chex.ArrayDevice
    forcing_phase: chex.ArrayDevice
    forcing_period: chex.ArrayDevice


# --- Extended ODE Params ---
@chex.dataclass
class SEIRS_ODEParams(AbstractODEParams):
    beta: chex.ArrayDevice
    gamma: chex.ArrayDevice
    sigma: chex.ArrayDevice
    omega: chex.ArrayDevice
    seasonality_params: SeasonalityParams


# --- Seasonality helper ---
def seasonality(t, params: SeasonalityParams):
    return 1.0 + params.forcing_amp * jnp.sin(
        2 * jnp.pi * t / params.forcing_period + params.forcing_phase
    )


# --- SEIRS ODE with seasonal forcing ---
def seirs_ode_seasonal(t: float, state: CompartmentState, p: SEIRS_ODEParams):
    s, e, i, r = state
    N = s + e + i + r
    beta_t = p.beta * seasonality(t, p.seasonality_params)
    ds = -beta_t * s * i / N + p.omega * r
    de = beta_t * s * i / N - p.sigma * e
    di = p.sigma * e - p.gamma * i
    dr = p.gamma * i - p.omega * r
    return (ds, de, di, dr)


# --- ODE Params getter with seasonal params ---
def get_seirs_odeparams(
    config: SimulationConfig,
    forcing_amp=0.2,
    forcing_phase=0.0,
    forcing_period=365.0,
):
    strain: Strain = config.parameters.transmission_params.strains[0]
    beta = strain.r0 / strain.infectious_period
    gamma = 1.0 / strain.infectious_period
    sigma = 1.0 / config.parameters.transmission_params.latent_period
    omega = 1.0 / config.parameters.transmission_params.waning_period
    seasonality_params = SeasonalityParams(
        forcing_amp=forcing_amp,
        forcing_phase=forcing_phase,
        forcing_period=forcing_period,
    )
    return SEIRS_ODEParams(
        beta=beta,
        gamma=gamma,
        sigma=sigma,
        omega=omega,
        seasonality_params=seasonality_params,
    )


# --- Main ---
if __name__ == "__main__":
    config = get_config()
    ode_params = get_seirs_odeparams(
        config, forcing_amp=0.2, forcing_phase=0.0, forcing_period=365.0
    )
    sol = simulate(
        ode=seirs_ode_seasonal,
        duration_days=1500,
        initial_state=config.initializer.get_initial_state(),
        ode_parameters=ode_params,
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
    plt.title("SEIRS Model With Seasonal Forcing")
    plt.show()
