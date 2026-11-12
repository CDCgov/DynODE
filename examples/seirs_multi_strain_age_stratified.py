"""An example of a SEIRS model with age stratification and multiple competing strains.

What did we need to modify from the base SEIRS model in order to add age stratification and multiple strains?

- **Compartment Structure**: We added age stratification by defining compartments with dimensions for age and strain.
                             An additional Cumulative compartment was added to track total infections over time.
- **Initializer**: The initializer was the site of most change, because we now needed to
                    distribute individuals across our new dimensions. To do this we made some
                    simple demographics assumptions and related the distribution of initial infections across strains to their R0.

- **ODE Parameters**: The ODE parameters were updated to handle multiple strains,
                    including strain-specific transmission rates and waning periods.
                    An indexing structure was added to improve readability with the multi-dimensional state.

- **ODE Function**: The ODE function was modified to compute the force of infection for each strain and age group,
                    and to handle the dynamics of multiple strains. Using the newly ported indexing structure
                    to improve readability.
"""

from datetime import date
from types import SimpleNamespace

import chex
import jax
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
    infected_by_dimension = Dimension(
        name="strain", bins=[Bin(name=s.strain_name) for s in strains]
    )
    # S = fully susceptible so previous infections don't matter
    s = Compartment(name="s", dimensions=[dimension_age])
    e = Compartment(
        name="e", dimensions=[dimension_age, infected_by_dimension]
    )
    i = Compartment(
        name="i", dimensions=[dimension_age, infected_by_dimension]
    )
    r = Compartment(
        name="r", dimensions=[dimension_age, infected_by_dimension]
    )
    c = Compartment(
        name="c", dimensions=[dimension_age, infected_by_dimension]
    )  # Cumulative book-keeping compartment

    # Age x Age contact matrix
    contact_matrix = jnp.array([[0.7, 0.3], [0.3, 0.7]])
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
        self,
        config: SimulationConfig,
        s0_prop=0.99,
        i0_prop=0.01,
        **kwargs,
    ) -> CompartmentState:
        """A slightly more complex initializer for SEIRS with age stratification and multiple strains.

        Note
        ----
        This initializer assumes two age groups with a fixed distribution.

        This initializer distributions initial infections across strains according to their relative R0 values."""
        # 2 age groups, 3 strains
        age_demographics = jnp.array([0.75, 0.25])
        # s_0 = jnp.zeros(config.get_compartment("s").shape)
        # initial susceptibles have no prior infections
        s_0 = self.population_size * s0_prop * age_demographics
        # no initial exposed persons
        e_0 = jnp.zeros(config.get_compartment("e").shape)
        # initial infected persons are dispersed across strains according to r0 of each strain
        strain_initial_dominance = jnp.array(
            vectorize_objects(
                config.parameters.transmission_params.strains, target="r0"
            )
        )
        strain_initial_dominance = strain_initial_dominance / jnp.sum(
            strain_initial_dominance
        )  # Normalize to sum to 1
        # Set initial infected persons, distributed by strain_initial_dominance
        i_0 = (
            self.population_size
            * i0_prop
            * age_demographics[:, None]
            * strain_initial_dominance
        )
        r_0 = jnp.zeros(config.get_compartment("r").shape)
        c_0 = jnp.zeros(
            config.get_compartment("c").shape
        )  # Start cumulative at zero
        return (s_0, e_0, i_0, r_0, c_0)


# --- ODE Params ---
# identify certain parameters as static so jax does not compile them.
@chex.dataclass(static_keynames=("idx"))
class SEIRS_MultiStrain_ODEParams(AbstractODEParams):
    beta: chex.ArrayDevice  # shape = (num_strains,)
    gamma: chex.ArrayDevice  # shape = (num_strains,)
    sigma: chex.ArrayDevice  # shape = (num_strains,)
    omega: chex.ArrayDevice  # shape = (num_strains,)
    contact_matrix: chex.ArrayDevice  # shape = (age,age)
    idx: SimpleNamespace  # for indexing in the odes


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
        idx=config.idx,
    )


# --- ODE ---
@jax.jit
def seirs_multi_strain_ode(
    t: float, state: CompartmentState, p: SEIRS_MultiStrain_ODEParams
):
    # state: (s, e, i, r, c)
    # s: (2,)  (age,)
    # e, i, r, c: (2, 3)  (age, strain)
    s, e, i, r, c = state
    N_age = (
        s
        + jnp.sum(e, axis=p.idx.e.strain)
        + jnp.sum(i, axis=p.idx.i.strain)
        + jnp.sum(r, axis=p.idx.r.strain)
    )

    # Force of infection for each age and strain
    fois = jnp.zeros_like(i)
    for strain in range(3):
        infectious_prop = i[:, strain] / N_age.squeeze()  # (2,)
        foi = p.beta[strain] * (p.contact_matrix @ infectious_prop)  # (2,)
        fois = fois.at[:, strain].set(foi)

    # ds: (2,) - sum over all strains of new infections, plus return from R
    ds = -jnp.sum(fois * s[:, None], axis=1) + jnp.sum(p.omega * r, axis=1)
    # de, di, dr, dc: (2, 3)
    de = fois * s[:, None] - p.sigma * e
    di = p.sigma * e - p.gamma * i
    dr = p.gamma * i - p.omega * r
    dc = fois * s[:, None]  # Cumulative incidence

    return (ds, de, di, dr, dc)


# --- Run and Plot ---
if __name__ == "__main__":
    # test out exactly identical strains but with Strain C
    # having a slightly higher R0, meaning it will eventually dominate
    config = get_config(
        r0s=[2.4, 2.5, 2.8],
        infectious_periods=[7.0, 7.0, 7.0],
        latent_periods=[3.0, 3.0, 3.0],
        waning_periods=[60.0, 60.0, 60.0],
    )
    ode_params = get_odeparams(config)
    initial_state = config.initializer.get_initial_state(config)
    sol = simulate(
        ode=seirs_multi_strain_ode,
        duration_days=500,
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

    # plot infection incidence by taking diff of cumulative incidence
    plt.figure(figsize=(12, 8))
    for strain_idx, strain_label in enumerate(strain_labels):
        incidence = jnp.diff(
            jnp.sum(c[:, :, strain_idx], axis=1), axis=0, prepend=jnp.nan
        )
        # get max day of incidence for this strain
        plt.plot(
            t, incidence, label=f"Total Incidence (Strain {strain_label})"
        )
    plt.xlabel("Days")
    plt.ylabel("Infection Incidence")
    plt.legend()
    plt.title("Infection Incidence by Strain")
    plt.tight_layout()
    plt.show()
