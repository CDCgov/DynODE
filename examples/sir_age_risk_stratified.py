from datetime import date

import chex
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from pydantic import ConfigDict, Field

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
from dynode.config.bins import AgeBin, Bin
from dynode.infer import sample_then_resolve
from dynode.simulation import AbstractODEParams, simulate
from dynode.typing import CompartmentGradients, CompartmentState


class SIRInitializer(Initializer):
    """Initializer for SIR model, setting initial conditions for compartments."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    description: str = "An SIR initializer"
    initialize_date: date = date(2022, 2, 11)  # random date
    population_size: int = 1000

    age_demographics: jnp.ndarray = Field(...)
    risk_prop: jnp.ndarray = Field(...)

    def get_initial_state(
        self, s0_prop=0.99, i0_prop=0.01
    ) -> CompartmentState:
        """Get initial compartment values for an SIR model stratified by age."""
        assert s0_prop + i0_prop == 1.0, (
            "s0_prop and i0_prop must sum to 1.0, "
            f"got {s0_prop} and {i0_prop}."
        )

        age_risk_prop = (
            self.age_demographics[:, None] * self.risk_prop
        )  # age by row, risk by column
        num_susceptibles = self.population_size * s0_prop
        s_0 = num_susceptibles * age_risk_prop
        num_infectious = self.population_size * i0_prop
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
    age_contact_matrix=jnp.array(
        [[0.7, 0.2, 0.1], [0.2, 0.7, 0.1], [0.1, 0.1, 0.8]]
    ),
    risk_contact_matrix=jnp.array([[0.7, 0.3], [0.3, 0.7]]),
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
        name="risk", bins=[Bin(name="high"), Bin(name="low")]
    )

    assert len(age_demographics) == len(age_dimension), (
        "Length of age proportions must match the number of age bins."
    )

    assert_risk_prop_shape(
        risk_prop=risk_prop,
        n_age=len(age_dimension),
        n_risk=len(risk_dimension),
    )

    assert_square_matrix(
        x=age_contact_matrix, n=len(age_dimension), name="Age contact matrix"
    )
    assert_square_matrix(
        x=risk_contact_matrix,
        n=len(risk_dimension),
        name="Risk contact matrix",
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

    # Contact matrix by age and risk:
    contact_matrix = jnp.einsum(
        "ij, kl -> ikjl", age_contact_matrix, risk_contact_matrix
    )

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


def assert_square_matrix(
    x: jnp.ndarray,
    n: int,
    name: str,
):
    """
    Check if x is a 2D square matrix of more than one bin exists for the dimension.
    Only pass when more than one bin exists.

    Args:
    x: the array to check
    n: number of bins in the dimension
    name: name of the dimension
    vector_shape: the expected shape for the vector/matrix
    """

    if n > 1:
        assert x.ndim == 2 and x.shape == (n, n), (
            f"{name} must be a square 2D array of shape ({n}, {n})."
        )
    else:
        raise ValueError(f"{name} dimension must have at least two bins.")


def assert_risk_prop_shape(
    risk_prop: jnp.ndarray,
    n_age: int,
    n_risk: int,
):
    """
    Check if risk_prop has the correct shape based on number of age and risk bins.
    Only pass when more than one bin exists for the risk dimension.
    """

    if n_risk > 1:
        assert risk_prop.ndim == 2 and risk_prop.shape == (n_age, n_risk), (
            f"Risk proportions must be 2D with shape ({n_age}, {n_risk}) when there are multiple risk bins."
        )
    else:
        raise ValueError("Risk dimension must have at least one bin.")


# define the behavior of the ODEs and the parameters they take
@chex.dataclass()
class SIR_ODEParams(AbstractODEParams):
    beta: chex.ArrayDevice  # r0/infectious period
    gamma: chex.ArrayDevice  # 1/infectious period
    contact_matrix: chex.ArrayDevice  # contact matrix


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
        "ijkl,ij -> ij", (p.contact_matrix) / pop_size, i
    )

    s_to_i = jnp.einsum("ij,ij -> ij", s, force_of_infection)
    i_to_r = i * p.gamma
    ds = -s_to_i
    di = s_to_i - i_to_r
    dr = i_to_r
    return tuple([ds, di, dr])


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
