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

    s0_prop: jnp.ndarray | float = Field(...)
    i0_prop: jnp.ndarray | float = Field(...)

    def get_initial_state(self) -> CompartmentState:
        assert (self.s0_prop + self.i0_prop == 1.0).all(), (
            "each group in s0_prop and i0_prop must sum to 1.0, "
            f"got {self.s0_prop} and {self.i0_prop}."
        )

        age_risk_prop = (
            self.age_demographics[:, None] * self.risk_prop
        )  # age by row, risk by column
        num_susceptibles = self.population_size * self.s0_prop
        s_0 = num_susceptibles * age_risk_prop
        num_infectious = self.population_size * self.i0_prop
        i_0 = num_infectious * age_risk_prop
        r_0 = jnp.zeros(s_0.shape)

        # SimulationConfig has no impact on initial state in this example
        return (
            s_0,
            i_0,
            r_0,
        )


def get_config(config_params: dict) -> SimulationConfig:
    """Create a SimulationConfig for an age-stratified SIR model."""
    r_0 = config_params["r_0"]
    infectious_period = config_params["infectious_period"]

    age_demographics = config_params["age_demographics"]
    risk_prop = config_params["risk_prop"]

    age_contact_matrix = config_params["age_contact_matrix"]
    risk_contact_matrix = config_params["risk_contact_matrix"]

    age_dimension = config_params["age_dimension"]
    risk_dimension = config_params["risk_dimension"]

    s0_prop = config_params["s0_prop"]
    i0_prop = config_params["i0_prop"]

    assert len(age_demographics) == len(age_dimension)
    assert risk_prop.shape[1] == len(
        risk_dimension
    )  # for each age group, what is the risk prop?

    assert s0_prop.shape[0] == len(age_dimension)
    assert s0_prop.shape[1] == len(risk_dimension)

    assert i0_prop.shape[0] == len(age_dimension)
    assert i0_prop.shape[1] == len(risk_dimension)

    assert age_contact_matrix.shape[0] == age_contact_matrix.shape[1]
    assert age_contact_matrix.shape[0] == len(age_dimension)

    assert risk_contact_matrix.shape[0] == risk_contact_matrix.shape[1]
    assert risk_contact_matrix.shape[0] == len(risk_dimension)

    s = Compartment(name="s", dimensions=[age_dimension, risk_dimension])
    i = Compartment(name="i", dimensions=[age_dimension, risk_dimension])
    r = Compartment(name="r", dimensions=[age_dimension, risk_dimension])

    strain = [
        Strain(strain_name="swo9", r0=r_0, infectious_period=infectious_period)
    ]

    initializer = SIRInitializer(
        age_demographics=age_demographics,
        risk_prop=risk_prop,
        s0_prop=s0_prop,
        i0_prop=i0_prop,
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
        "ijkl,ij -> kl", (p.contact_matrix) / pop_size, i
    )

    s_to_i = jnp.einsum("ij,ij -> ij", s, force_of_infection)
    i_to_r = i * p.gamma
    ds = -s_to_i
    di = s_to_i - i_to_r
    dr = i_to_r
    return tuple([ds, di, dr])


if __name__ == "__main__":
    config_params = {
        "r_0": 2.0,
        "infectious_period": 7.0,
        "s0_prop": jnp.array([[0.99, 1.0], [0.99, 0.99], [1.0, 1.0]]),
        "i0_prop": jnp.array([[0.01, 0.0], [0.01, 0.01], [0.0, 0.0]]),
        "age_demographics": jnp.array([0.7, 0.2, 0.1]),
        "risk_prop": jnp.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]),
        "age_contact_matrix": jnp.array(
            [[0.8, 0.2, 0.0], [0.2, 0.8, 0.0], [0.0, 0.0, 1.0]]
        ),
        "risk_contact_matrix": jnp.array([[0.5, 0.5], [0.5, 0.5]]),
        "age_dimension": Dimension(
            name="age",
            bins=[
                AgeBin(0, 17, "young"),
                AgeBin(18, 64, "adult"),
                AgeBin(65, 99, "elderly"),
            ],
        ),
        "risk_dimension": Dimension(
            name="risk", bins=[Bin(name="high"), Bin(name="low")]
        ),
    }
    config = get_config(config_params=config_params)

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
    age_risk_dim = len(age_labels) * len(risk_labels)

    age_risk_labels = [f"{a} {b}" for a in age_labels for b in risk_labels]

    for idx, label in enumerate(age_risk_labels):
        plt.plot(
            t,
            s.reshape(-1, age_risk_dim)[:, idx],
            label=f"Susceptible ({label})",
        )
        # plt.plot(
        #     t,
        #     i.reshape(-1, age_risk_dim)[:, idx],
        #     label=f"Infectious ({label})",
        # )
        # plt.plot(
        #     t,
        #     r.reshape(-1, age_risk_dim)[:, idx],
        #     label=f"Recovered ({label})",
        # )
    plt.xlabel("Days")
    plt.ylabel("Population")
    plt.legend(ncol=1, bbox_to_anchor=(1.0, 1.05))
    plt.title("Simple SIR Model (Age and Risk Stratified)")
    plt.savefig("sir_age_risk_stratified.png", bbox_inches="tight")
