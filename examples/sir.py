# An Example of how to simulate a basic SIR compartmental model using the dynode package
# Including all the class setup
# %% imports, mostly for class creation
from datetime import date

import chex
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpyro
from diffrax import Solution, is_okay

from dynode.model_configuration import (
    Bin,
    Compartment,
    Dimension,
    InferenceProcess,
    Initializer,
    MCMCParams,
    Params,
    SimulationConfig,
    SolverParams,
    Strain,
    SVIParams,
    TransmissionParams,
)
from dynode.odes import AbstractODEParams, simulate
from dynode.sample import sample_then_resolve
from dynode.typing import CompartmentGradients, CompartmentState


# %% class definitions
class SIRInitializer(Initializer):
    """Initializer for SIR model, setting initial conditions for compartments."""

    def __init__(self):
        super().__init__(
            description="An SIR initalizer",
            initialize_date=date(2025, 3, 13),
            population_size=1,
        )

    def get_initial_state(self, **kwargs):
        _: SimulationConfig = kwargs["SIRConfig"]
        # SimulationConfig has no impact on initial state in this example
        return (jnp.array([0.99]), jnp.array([0.01]), jnp.array([0.00]))


class SIRConfig(SimulationConfig):
    def __init__(self):
        """Set parameters for an SIR compartmental model.

        This includes compartment shape, initializer, and solver/transmission parameters."""
        dimension = Dimension(name="value", bins=[Bin(name="value")])
        s = Compartment(name="s", dimensions=[dimension])
        i = Compartment(name="i", dimensions=[dimension])
        r = Compartment(name="r", dimensions=[dimension])
        strain = [
            Strain(strain_name="example_strain", r0=2.0, infectious_period=7.0)
        ]
        parameters = Params(
            solver_params=SolverParams(),
            transmission_params=TransmissionParams(
                strains=strain,
                strain_interactions={
                    "example_strain": {"example_strain": 1.0}
                },
            ),
        )
        super().__init__(
            compartments=[s, i, r],
            initializer=SIRInitializer(),
            parameters=parameters,
        )


@chex.dataclass
class SIR_ODEParams(AbstractODEParams):
    beta: chex.ArrayDevice  # r0/infectious period
    gamma: chex.ArrayDevice  # 1/infectious period
    pass


def get_odeparams(transmission_params: TransmissionParams) -> SIR_ODEParams:
    """Transform and vectorize transmission parameters into ODE parameters."""
    transmission_params = sample_then_resolve(transmission_params)
    strain = transmission_params.strains[0]
    assert isinstance(strain.r0, float)
    beta = strain.r0 / strain.infectious_period
    gamma = 1 / strain.infectious_period
    return SIR_ODEParams(beta=jnp.array(beta), gamma=jnp.array(gamma))


@jax.jit
def sir_ode(
    t: float, state: CompartmentState, p: SIR_ODEParams
) -> CompartmentGradients:
    s, i, _ = state
    s_to_i = p.beta * s * i
    i_to_r = i * p.gamma
    ds = -s_to_i
    di = s_to_i - i_to_r
    dr = i_to_r
    return tuple([ds, di, dr])


# %% simulation

# set up config
config = SIRConfig()


def model(config: SimulationConfig, obs_data: jax.Array = None):
    ode_params = get_odeparams(config.parameters.transmission_params)

    # we need just the jax arrays for the initial state to the ODEs
    initial_state = config.initializer.get_initial_state(SIRConfig=config)
    # solve the odes for 100 days

    solution: Solution = simulate(
        ode=sir_ode,
        duration_days=100,
        initial_state=initial_state,
        ode_parameters=ode_params,
        solver_parameters=config.parameters.solver_params,
    )
    # step c compare to obs data
    if obs_data is not None:
        incidence = jnp.diff(solution.ys[1].flatten())
        incidence = jnp.maximum(incidence, 1e-6)
        numpyro.sample(
            "inf_incidence",
            numpyro.distributions.Poisson(incidence),
            obs=obs_data,
        )
    return solution


solution = model(config)

if is_okay(solution.result):
    print("solution is okay")
    assert solution.ys is not None
    plt.plot(solution.ys[0], label="s")
    plt.plot(solution.ys[1], label="i")
    plt.plot(solution.ys[2], label="r")
    plt.legend()
    plt.show()

# now lets replace this strain with one with a prior at infectious_period instead
config.parameters.transmission_params.strains = [
    Strain(
        strain_name="example_strain",
        r0=2.0,
        infectious_period=numpyro.distributions.TruncatedNormal(
            loc=8, scale=2, low=2, high=15
        ),
    )
]

inference_process = InferenceProcess(
    simulator=model,
    inference_method=numpyro.infer.MCMC,
    inference_parameters=MCMCParams(
        num_warmup=1000, num_samples=1000, num_chains=1, nuts_max_tree_depth=10
    ),
)
inference_process_svi = InferenceProcess(
    simulator=model,
    inference_method=numpyro.infer.MCMC,
    inference_parameters=SVIParams(),
)
incidence = jnp.diff(solution.ys[1].flatten())
print(incidence.shape)
inferer = inference_process.infer(config=config, obs_data=incidence)
posterior_samples = inferer.get_samples()
print(
    f"Recovered infectious period: {jnp.mean(posterior_samples['strains_0_infectious_period'])}"
)
# %%
