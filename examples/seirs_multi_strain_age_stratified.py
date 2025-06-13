"""An example of a SEIRS model with age stratification and multiple competing strains."""

from datetime import date

import chex
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from dynode.config import (
    Bin,
    Compartment,
    Dimension,
    Initializer,
    LastStrainImmuneHistoryDimension,
    Params,
    SimulationConfig,
    SolverParams,
    Strain,
    TransmissionParams,
)
from dynode.simulation import AbstractODEParams, simulate
from dynode.typing import CompartmentState
from dynode.utils import vectorize_objects


# --- Config ---
def get_config(
    r0s=(2.0, 2.5, 1.8),
    infectious_periods=(7.0, 6.0, 8.0),
    latent_periods=(3.0, 2.5, 4.0),
    waning_periods=(60.0, 80.0, 50.0),
) -> SimulationConfig:
    strains = [
        Strain(
            strain_name="A",
            r0=r0s[0],
            infectious_period=infectious_periods[0],
            exposed_to_infectious=latent_periods[0],
        ),
        Strain(
            strain_name="B",
            r0=r0s[1],
            infectious_period=infectious_periods[1],
            exposed_to_infectious=latent_periods[1],
        ),
        Strain(
            strain_name="C",
            r0=r0s[2],
            infectious_period=infectious_periods[2],
            exposed_to_infectious=latent_periods[2],
        ),
    ]
    dimension_age = Dimension(
        name="age", bins=[Bin(name="young"), Bin(name="old")]
    )
    # immune history only tracks last recovered from strain
    # use FullStratifiedImmuneHistoryDimension for full history tracking
    dimension_strain = LastStrainImmuneHistoryDimension(strains=strains)
    s = Compartment(name="s", dimensions=[dimension_age, dimension_strain])
    e = Compartment(name="e", dimensions=[dimension_age, dimension_strain])
    i = Compartment(name="i", dimensions=[dimension_age, dimension_strain])
    r = Compartment(name="r", dimensions=[dimension_age, dimension_strain])
    c = Compartment(
        name="c", dimensions=[dimension_age, dimension_strain]
    )  # Cumulative

    # Age x Age contact matrix
    contact_matrix = jnp.array([[0.7, 0.3], [0.3, 0.7]])
    contact_matrix = contact_matrix / jnp.max(
        jnp.real(jnp.linalg.eigvals(contact_matrix))
    )
    # Strain interaction matrix (no cross-immunity, all compete equally)
    strain_names = ["A", "B", "C"]
    strain_interactions = {
        s1: {s2: 1.0 for s2 in strain_names} for s1 in strain_names
    }

    transmission_params = TransmissionParams(
        strains=strains,
        strain_interactions=strain_interactions,
        contact_matrix=contact_matrix,
        waning_period=waning_periods,  # tuple, one per strain R -> S
    )

    parameters = Params(
        solver_params=SolverParams(),
        transmission_params=transmission_params,
    )

    config = SimulationConfig(
        compartments=[s, e, i, r, c],  # Add c here
        initializer=SEIRSStratifiedInitializer(),
        parameters=parameters,
    )
    return config


# --- Initializer for SEIRS with age stratification and multiple strains ---
class SEIRSStratifiedInitializer(Initializer):
    def __init__(self, population_size=1000):
        super().__init__(
            description="SEIRS initializer with age stratification",
            initialize_date=date(2022, 2, 11),
            population_size=population_size,
        )

    def get_initial_state(
        self, s0_prop=0.97, e0_prop=0.0, i0_prop=0.03, r0_prop=0.0, **kwargs
    ) -> CompartmentState:
        # 2 age groups, 3 strains
        age_demographics = jnp.array([0.75, 0.25])
        shape = (2, 3)
        s_0 = (
            self.population_size
            * s0_prop
            * age_demographics[:, None]
            * jnp.ones(shape)
        )
        e_0 = (
            self.population_size
            * e0_prop
            * age_demographics[:, None]
            * jnp.ones(shape)
        )
        i_0 = (
            self.population_size
            * i0_prop
            * age_demographics[:, None]
            * jnp.ones(shape)
        )
        r_0 = (
            self.population_size
            * r0_prop
            * age_demographics[:, None]
            * jnp.ones(shape)
        )
        c_0 = jnp.zeros(shape)  # Start cumulative at zero
        return (s_0, e_0, i_0, r_0, c_0)


# --- ODE Params ---
@chex.dataclass
class SEIRS_MultiStrain_ODEParams(AbstractODEParams):
    beta: chex.ArrayDevice  # (num_strains,)
    gamma: chex.ArrayDevice  # (num_strains,)
    sigma: chex.ArrayDevice  # (num_strains,)
    omega: chex.ArrayDevice  # (num_strains,)
    contact_matrix: chex.ArrayDevice  # (age,age)


def get_odeparams(config: SimulationConfig) -> SEIRS_MultiStrain_ODEParams:
    tp = config.parameters.transmission_params
    # pull out parameters for each strain
    r0s = jnp.array(vectorize_objects(tp.strains, target="r0"))
    infectious_periods = jnp.array(
        vectorize_objects(tp.strains, target="infectious_period")
    )
    exposed_to_infectious = jnp.array(
        vectorize_objects(tp.strains, target="exposed_to_infectious")
    )
    betas = r0s / infectious_periods  # (3,)
    sigmas = 1.0 / exposed_to_infectious  # (3,)
    gammas = 1.0 / infectious_periods  # (3,)
    omegas = 1.0 / jnp.array(tp.waning_period)  # (3,)

    return SEIRS_MultiStrain_ODEParams(
        beta=betas,
        gamma=gammas,
        sigma=sigmas,
        omega=omegas,
        contact_matrix=tp.contact_matrix,
    )


# --- ODE ---
@jax.jit
def seirs_multi_strain_ode(
    t: float, state: CompartmentState, p: SEIRS_MultiStrain_ODEParams
):
    # state: (s, e, i, r, c), each (2, 3): (age, strain)
    s, e, i, r, c = state
    N = s + e + i + r  # (2, 3)
    N_age = jnp.sum(N, axis=1, keepdims=True)  # (2, 1)
    # Force of infection for each age and strain
    # For each age a and strain k:
    #   lambda[a, k] = beta[k] * sum_{b} contact[a, b] * i[b, k] / N_age[b]
    # (2, 3)
    fois = jnp.zeros_like(s)
    for strain in range(3):
        foi = jnp.sum(
            p.contact_matrix * (i[:, strain] / N_age.squeeze()), axis=1
        )
        fois = fois.at[:, strain].set(p.beta[strain] * foi)
    # ODEs
    ds = -fois * s + p.omega * r
    de = fois * s - p.sigma * e
    di = p.sigma * e - p.gamma * i
    dr = p.gamma * i - p.omega * r
    dc = fois * s  # Cumulative incidence
    return (ds, de, di, dr, dc)


@jax.jit
def seirs_multi_strain_ode2(
    t: float, state: CompartmentState, p: SEIRS_MultiStrain_ODEParams
):
    # state: (s, e, i, r, c), each (2, 3): (age, strain)
    s, e, i, r, c = state
    N = s + e + i + r  # (2, 3)
    N_age = jnp.sum(N, axis=1, keepdims=True)  # (2, 1)
    # Force of infection for each age and strain
    # For each age a and strain k:
    #   lambda[a, k] = beta[k] * sum_{b} contact[a, b] * i[b, k] / N_age[b]
    # (2, 3)
    fois = jnp.zeros_like(s)
    for strain in range(3):
        # Compute proportion infectious in each age group
        infectious_prop = i[:, strain] / N_age.squeeze()  # (2,)
        # Matrix multiplication: contact_matrix @ infectious_prop
        foi = p.contact_matrix @ infectious_prop  # (2,)
        fois = fois.at[:, strain].set(p.beta[strain] * foi)
    # ODEs
    ds = -fois * s + p.omega * r
    de = fois * s - p.sigma * e
    di = p.sigma * e - p.gamma * i
    dr = p.gamma * i - p.omega * r
    dc = fois * s  # Cumulative incidence
    return (ds, de, di, dr, dc)


# --- Run and Plot ---
if __name__ == "__main__":
    # TODO this is still broken somehow, need to fix the ODEs
    config = get_config()
    ode_params = get_odeparams(config)
    initial_state = config.initializer.get_initial_state()
    sol = simulate(
        ode=seirs_multi_strain_ode2,
        duration_days=300,
        initial_state=initial_state,
        ode_parameters=ode_params,
        solver_parameters=config.parameters.solver_params,
    )

    s, e, i, r, c = sol.ys  # each (timesteps, 2, 3)
    t = sol.ts
    age_labels = ["Young", "Old"]
    strain_labels = ["A", "B", "C"]

    # Plot cumulative incidence for each strain, summed over age groups
    plt.figure(figsize=(12, 8))
    for strain_idx, strain_label in enumerate(strain_labels):
        plt.plot(
            t,
            jnp.sum(c[:, :, strain_idx], axis=1),
            label=f"Cumulative ({strain_label})",
        )
    plt.xlabel("Days")
    plt.ylabel("Cumulative Incidence")
    plt.legend()
    plt.title("Cumulative Incidence by Strain")
    plt.tight_layout()
    plt.show()

    # Assert population conservation: sum of derivatives should be zero
    total_change = jnp.sum(s + e + i + r, axis=(1, 2))
    assert jnp.allclose(total_change, total_change[0]), (
        "Population not conserved across time steps"
    )

    # plot infection incidence by taking diff of cumulative incidence
    plt.figure(figsize=(12, 8))
    for strain_idx, strain_label in enumerate(strain_labels):
        incidence = jnp.diff(
            jnp.sum(c[:, :, strain_idx], axis=1), axis=0, prepend=0
        )
        plt.plot(
            t, incidence, label=f"Total Incidence (Strain {strain_label})"
        )
    plt.xlabel("Days")
    plt.ylabel("Infection Incidence")
    plt.legend()
    plt.title("Infection Incidence by Strain")
    plt.tight_layout()
    plt.show()
