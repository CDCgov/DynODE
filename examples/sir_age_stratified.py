# An Example of how to simulate a basic SIR compartmental model using the dynode package
# Including all the class setup
# %% imports and definitions
# most of these imports are for type hinting
from datetime import date
from types import SimpleNamespace

import chex
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from diffrax import Solution

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
from dynode.infer import sample_then_resolve
from dynode.simulation import AbstractODEParams, simulate
from dynode.typing import CompartmentGradients, CompartmentState

SHOW_PLOTS = False  # turn to true for some additional insights!


# --- SIR Initializer with age stratification ---
class SIRInitializer(Initializer):
    """Initializer for SIR model, setting initial conditions for compartments."""

    def __init__(self):
        """Create an SIR Initializer."""
        super().__init__(
            description="An SIR initalizer",
            initialize_date=date(2022, 2, 11),  # random date
            population_size=1000,
        )

    def get_initial_state(
        self, s0_prop=0.99, i0_prop=0.01, **kwargs
    ) -> CompartmentState:
        """Get initial compartment values for an SIR model stratified by age."""
        assert s0_prop + i0_prop == 1.0, (
            "s0_prop and i0_prop must sum to 1.0, "
            f"got {s0_prop} and {i0_prop}."
        )
        # proportion of young to old in the population
        age_demographics = jnp.array([0.75, 0.25])
        num_susceptibles = self.population_size * jnp.array([s0_prop])
        s_0 = num_susceptibles * age_demographics
        num_infectious = self.population_size * jnp.array([i0_prop])
        i_0 = num_infectious * age_demographics
        r_0 = jnp.array([0.0, 0.0])
        # SimulationConfig has no impact on initial state in this example
        return (
            s_0,
            i_0,
            r_0,
        )


# --- SIRConfig for bin definitions and strain specification---
class SIRConfig(SimulationConfig):
    """A static SIR config class with basic age structure."""

    def __init__(self):
        """Set parameters for a static SIR compartmental model.

        This includes compartment shape, initializer, and solver/transmission parameters.
        """
        dimension = Dimension(
            name="age", bins=[Bin(name="young"), Bin(name="old")]
        )
        s = Compartment(name="s", dimensions=[dimension])
        i = Compartment(name="i", dimensions=[dimension])
        r = Compartment(name="r", dimensions=[dimension])
        strain = [Strain(strain_name="swo9", r0=2, infectious_period=7.0)]
        contact_matrix = jnp.array([[0.7, 0.3], [0.3, 0.7]])
        # normalize contact matrix by the spectral radius
        contact_matrix = contact_matrix / jnp.max(
            jnp.real(jnp.linalg.eigvals(contact_matrix))
        )
        parameters = Params(
            solver_params=SolverParams(),
            transmission_params=TransmissionParams(
                strains=strain,
                strain_interactions={"swo9": {"swo9": 1.0}},
                # contact matrix for young/old interactions
                contact_matrix=contact_matrix,
            ),
        )
        super().__init__(
            compartments=[s, i, r],
            initializer=SIRInitializer(),
            parameters=parameters,
        )


# define the behavior of the ODEs and the parameters they take
@chex.dataclass(static_keynames=["idx"])
class SIR_ODEParams(AbstractODEParams):
    beta: chex.ArrayDevice  # r0/infectious period
    gamma: chex.ArrayDevice  # 1/infectious period
    contact_matrix: chex.ArrayDevice  # contact matrix
    idx: SimpleNamespace  # indexing object for the compartments


# define a function to easily translate the object oriented TransmissionParams
# into the vectorized ODEParams.
def get_odeparams(config: SimulationConfig) -> SIR_ODEParams:
    """Transform and vectorize transmission parameters into ODE parameters."""
    transmission_params = sample_then_resolve(
        config.parameters.transmission_params
    )
    strain = transmission_params.strains[0]
    beta = strain.r0 / strain.infectious_period  # type: ignore
    gamma = 1 / strain.infectious_period
    return SIR_ODEParams(
        beta=jnp.array(beta),
        gamma=jnp.array(gamma),
        contact_matrix=transmission_params.contact_matrix,
        idx=config.idx,
    )


@jax.jit
def sir_ode(
    t: float, state: CompartmentState, p: SIR_ODEParams
) -> CompartmentGradients:
    """A simple SIR ODE model with no time-varying components."""
    s, i, r = state
    pop_size = s + i + r
    force_of_infection = p.beta * jnp.sum(
        (p.contact_matrix * i) / pop_size, axis=1
    )
    s_to_i = s * force_of_infection
    i_to_r = i * p.gamma
    ds = -s_to_i
    di = s_to_i - i_to_r
    dr = i_to_r
    return tuple([ds, di, dr])


# %% setup simulation process.
# instantiate the config


def run_simulation(config: SimulationConfig, tf) -> Solution:
    ode_params = get_odeparams(config)

    # we need just the jax arrays for the initial state to the ODEs
    initial_state = config.initializer.get_initial_state(SIRConfig=config)
    # solve the odes for 100 days
    solution: Solution = simulate(
        ode=sir_ode,
        duration_days=tf,
        initial_state=initial_state,
        ode_parameters=ode_params,
        solver_parameters=config.parameters.solver_params,
    )
    return solution


# --- Run and Plot ---
if __name__ == "__main__":
    config = SIRConfig()

    sol = simulate(
        ode=sir_ode,
        duration_days=150,
        initial_state=config.initializer.get_initial_state(),
        ode_parameters=get_odeparams(config),
        solver_parameters=config.parameters.solver_params,
    )
    # Plot S, I, R for each age group separately
    age_labels = ["Young", "Old"]
    s, i, r = sol.ys  # each is (timesteps, 2)
    t = sol.ts

    for idx, label in enumerate(age_labels):
        plt.plot(t, s[:, idx], label=f"Susceptible ({label})")
        plt.plot(t, i[:, idx], label=f"Infectious ({label})")
        plt.plot(t, r[:, idx], label=f"Recovered ({label})")
    plt.xlabel("Days")
    plt.ylabel("Population")
    plt.legend()
    plt.title("Simple SIR Model (Age Stratified)")
    plt.show()
