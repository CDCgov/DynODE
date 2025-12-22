from datetime import date
from typing import Annotated, Any

import chex
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from diffrax import Solution
from pydantic import BeforeValidator, ConfigDict, Field, field_validator

# weird
# unit test, sci eval, integration testing
from dynode.config import (
    Compartment,
    Dimension,
    Initializer,
    Params,
    SimulationConfig,
    SolverParams,
    Strain,
    TransmissionParams,
)
from dynode.config.bins import AgeBin, RiskBin
from dynode.infer import sample_then_resolve
from dynode.simulation import AbstractODEParams, simulate
from dynode.typing import CompartmentGradients, CompartmentState


def to_jax_array(v):
    return jnp.array(v)


JaxArray = Annotated[jnp.ndarray, BeforeValidator(to_jax_array)]


class SIRInitializer(Initializer):
    """Initializer for SIR model, setting initial conditions for compartments."""

    # Tell Pydantic it's okay to have jax.Array/jnp.ndarray fields
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    # Override parent defaults (instead of doing this in __init__)
    description: str = "An SIR initializer"
    initialize_date: date = date(2022, 2, 11)  # random date
    population_size: int = 1000

    # New fields as jnp.ndarray
    age_demographics: jnp.ndarray = Field(...)
    risk_prop: jnp.ndarray = Field(...)

    # Convert input to jnp.array before validation
    @field_validator("age_demographics", "risk_prop", mode="before")
    @classmethod
    def _to_jax_array(cls, v: Any) -> jnp.ndarray:
        return jnp.array(v)

    def get_initial_state(
        self, s0_prop=0.99, i0_prop=0.01, **kwargs
    ) -> CompartmentState:
        """Get initial compartment values for an SIR model stratified by age."""
        assert s0_prop + i0_prop == 1.0, (
            "s0_prop and i0_prop must sum to 1.0, "
            f"got {s0_prop} and {i0_prop}."
        )

        age_risk_prop = (
            self.age_demographics[:, None] * self.risk_prop
        )  # age by row, risk by column
        num_susceptibles = self.population_size * jnp.array([s0_prop])
        s_0 = num_susceptibles * age_risk_prop
        num_infectious = self.population_size * jnp.array([i0_prop])
        i_0 = num_infectious * age_risk_prop
        r_0 = jnp.zeros(s_0.shape)

        # SimulationConfig has no impact on initial state in this example
        return (
            s_0,
            i_0,
            r_0,
        )


def get_config(
    r_0=2.0,
    infectious_period=7.0,
    age_demographics=jnp.array([0.7, 0.2, 0.1]),
    risk_prop=jnp.array([[0.1, 0.9], [0.6, 0.4], [0.8, 0.2]]),
) -> SimulationConfig:
    """Create a SimulationConfig for an age-stratified SIR model."""
    age_dimension = Dimension(
        name="age",
        bins=[
            AgeBin(0, 17, name="young"),
            AgeBin(18, 64, name="adult"),
            AgeBin(65, 99, name="elderly"),
        ],
    )
    risk_dimension = Dimension(
        name="risk", bins=[RiskBin(name="high"), RiskBin(name="low")]
    )

    s = Compartment(name="s", dimensions=[age_dimension, risk_dimension])
    i = Compartment(name="i", dimensions=[age_dimension, risk_dimension])
    r = Compartment(name="r", dimensions=[age_dimension, risk_dimension])

    strain = [
        Strain(strain_name="swo9", r0=r_0, infectious_period=infectious_period)
    ]

    initializer = SIRInitializer(
        age_demographics=age_demographics, risk_prop=risk_prop
    )
    # contact_matrix = jnp.array([[[[0.25, 0.15], [0.35, 0.25]], [[0.15,0.35], [0.1,0.4]]],
    #                             [[[0.4,0.1],[0.2,0.3]],[[0.25,0.35],[0.3,0.1]]]])

    age_risk_prop = (
        initializer.age_demographics[:, None] * initializer.risk_prop
    )  # age by row, risk by column
    contact_matrix = (
        age_risk_prop[:, :, None, None] * age_risk_prop[None, None, :, :]
    )

    check_dimension(contact_matrix, age_demographics, risk_prop)

    # normalize contact matrix by the spectral radius
    # contact_matrix = contact_matrix / jnp.max(
    #     jnp.real(jnp.linalg.eigvals(contact_matrix))
    # )

    parameters = Params(
        solver_params=SolverParams(),
        transmission_params=TransmissionParams(
            strains=strain,
            strain_interactions={"swo9": {"swo9": 1.0}},
            contact_matrix=contact_matrix,
        ),
    )
    config = SimulationConfig(
        compartments=[s, i, r],
        initializer=initializer,
        parameters=parameters,
    )

    return config


def check_dimension(contact_matrix, age_demographics, risk_prop):
    assert contact_matrix.shape == (
        age_demographics.shape[0],
        risk_prop.shape[1],
        risk_prop.shape[0],
        risk_prop.shape[1],
    ), "Contact matrix shape does not match age demographics."


# define the behavior of the ODEs and the parameters they take
@chex.dataclass()
class SIR_ODEParams(AbstractODEParams):
    beta: chex.ArrayDevice  # r0/infectious period
    gamma: chex.ArrayDevice  # 1/infectious period
    contact_matrix: chex.ArrayDevice  # contact matrix


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
    )


@jax.jit
def sir_ode(
    t: float, state: CompartmentState, p: SIR_ODEParams
) -> CompartmentGradients:
    """A simple SIR ODE model with no time-varying components."""
    s, i, r = state
    pop_size = s + i + r
    force_of_infection = p.beta * jnp.einsum(
        "ijkl,mn -> ij", (p.contact_matrix) / pop_size, i
    )
    print(force_of_infection)
    s_to_i = s * force_of_infection
    i_to_r = i * p.gamma
    ds = -s_to_i
    di = s_to_i - i_to_r
    dr = i_to_r
    return tuple([ds, di, dr])


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
    config = get_config()

    sol = simulate(
        ode=sir_ode,
        duration_days=150,
        initial_state=config.initializer.get_initial_state(),
        ode_parameters=get_odeparams(config),
        solver_parameters=config.parameters.solver_params,
    )

    s, i, r = sol.ys  # each is (timesteps, 2)
    t = sol.ts

    age_labels = ["Young", "Adult", "Elderly"]
    risk_labels = ["High risk", "Low risk"]
    age_risk_labels = [f"{a} {b}" for a in age_labels for b in risk_labels]
    for idx, label in enumerate(age_risk_labels):
        plt.plot(t, s.reshape(-1, 6)[:, idx], label=f"Susceptible ({label})")
        plt.plot(t, i.reshape(-1, 6)[:, idx], label=f"Infectious ({label})")
        plt.plot(t, r.reshape(-1, 6)[:, idx], label=f"Recovered ({label})")
    plt.xlabel("Days")
    plt.ylabel("Population")
    plt.legend(ncol=1, bbox_to_anchor=(1.0, 1.05))
    plt.title("Simple SIR Model (Age and Risk Stratified)")
    plt.savefig("sir_age_risk_stratified.png", bbox_inches="tight")
