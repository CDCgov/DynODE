# An Example of how to simulate a basic SIR compartmental model using the dynode package
# Including all the class setup
# %% imports and definitions
# most of these imports are for type hinting
from types import SimpleNamespace

import arviz as az
import chex
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpyro
from diffrax import Solution
from numpyro.infer import Predictive
from numpyro.infer.svi import SVIRunResult

from dynode.config import (
    SimulationConfig,
    SIRConfig,
    SIRInferedConfig,
)
from dynode.infer import MCMCProcess, SVIProcess, sample_then_resolve
from dynode.simulate import AbstractODEParams, simulate
from dynode.typing import CompartmentGradients, CompartmentState


# define the behavior of the ODEs and the parameters they take
@chex.dataclass(static_keynames=["idx"])
class SIR_ODEParams(AbstractODEParams):
    beta: chex.ArrayDevice  # r0/infectious period
    gamma: chex.ArrayDevice  # 1/infectious period
    contact_matrix: chex.ArrayDevice  # contact matrix
    idx: SimpleNamespace  # indexing object for the compartments
    pass


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


# TODO add enums to SIR.py where applicable.
# define your Jit compiled ODE function
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
config_static = SIRConfig()


def run_simulation(config: SimulationConfig, tf) -> Solution:
    ode_params = get_odeparams(config)

    # we need just the jax arrays for the initial state to the ODEs
    initial_state = config.initializer.get_initial_state(SIRConfig=config)
    # solve the odes for 100 days
    # TODO, what if you dont jit the ode method?
    solution: Solution = simulate(
        ode=sir_ode,
        duration_days=tf,
        initial_state=initial_state,
        ode_parameters=ode_params,
        solver_parameters=config.parameters.solver_params,
    )
    return solution


# define the entire process of simulating incidence
def model(
    config: SimulationConfig,
    tf,
    obs_data: jax.Array | None = None,
    infer_mode=False,
):
    """Numpyro model for simulating infection incidence of an SIR model."""
    solution: Solution = run_simulation(config, tf)
    # compare to observed data if we have it
    if infer_mode:
        assert solution.ys is not None, "mypy assert"
        incidence = jnp.diff(
            solution.ys[config.idx.r], axis=0
        )  # leading time axis
        incidence = jnp.maximum(incidence, 1e-6)
        numpyro.sample(
            "inf_incidence",
            numpyro.distributions.Poisson(incidence),
            obs=obs_data,
        )
    return solution


# produce synthetic data with fixed r0 and infectious period
solution = run_simulation(config_static, tf=100)
# plot the soliution
assert solution.ys is not None
idx = config_static.idx
# add 1 to each axis to account for the leading time dimension in `solution`
plt.plot(
    jnp.sum(solution.ys[config_static.idx.s], axis=idx.s.age + 1),
    label="s",
)
plt.plot(
    jnp.sum(solution.ys[config_static.idx.i], axis=idx.i.age + 1),
    label="i",
)
plt.plot(
    jnp.sum(solution.ys[config_static.idx.r], axis=idx.r.age + 1),
    label="r",
)
plt.legend()
plt.show()
# diff recovered individuals to recover lagged incidence for each age group
incidence = jnp.diff(solution.ys[idx.r], axis=0)
# %%
# set up inference process
# now lets infer the parameters of this strain instead
config_infer = SIRInferedConfig()
# creating two InferenceProcesses, one for MCMC and one for SVI
inference_process_mcmc = MCMCProcess(
    numpyro_model=model,
    num_warmup=1000,
    num_samples=1000,
    num_chains=1,
    nuts_max_tree_depth=10,
)
inference_process_svi = SVIProcess(
    numpyro_model=model,
    num_iterations=2000,
    num_samples=1000,  # for posterior generation
)
# %%
# running inference
print("fitting MCMC")
inferer_mcmc = inference_process_mcmc.infer(
    config=config_infer, tf=100, obs_data=incidence, infer_mode=True
)
posterior_samples_mcmc = inference_process_mcmc.get_samples()
# %%
print("fitting SVI")
inferer_svi = inference_process_svi.infer(
    config=config_infer, tf=100, obs_data=incidence, infer_mode=True
)
posterior_samples_svi = inference_process_svi.get_samples()

# %%
# printing results of inference
print(
    f"Parameterized value of R0: {config_static.parameters.transmission_params.strains[0].r0} "
    f"Infectious Period: {config_static.parameters.transmission_params.strains[0].infectious_period}"
)
# notice the name of the posterior sample mimics the index of `transmission_params.strains`
# this will help you find parameters later on.
print(
    f"MCMC posterior's R0: {jnp.mean(posterior_samples_mcmc['strains_0_r0'])}, "
    f"Infectious Period: {jnp.mean(posterior_samples_mcmc['strains_0_infectious_period'])}"
)
print(
    f"SVI posterior's R0: {jnp.mean(posterior_samples_svi['strains_0_r0'])}, "
    f"Infectious Period: {jnp.mean(posterior_samples_svi['strains_0_infectious_period'])}"
)
# %%
mcmc_arviz = inference_process_mcmc.to_arviz()
svi_arviz = inference_process_svi.to_arviz()
# %%
axes = az.plot_density(
    [mcmc_arviz],
    data_labels=["R0"],
    var_names=["strains_0_r0"],
    shade=0.2,
)

fig = axes.flatten()[0].get_figure()
fig.suptitle("Density Interval for R0")

plt.show()
mcmc_arviz
# %%
svi_arviz
# %%
# projecting forward
# now lets turn on Predictive mode and do some projections forward without observed data
predictive_mcmc = Predictive(
    model,
    posterior_samples=posterior_samples_mcmc,
    exclude_deterministic=False,
)
posterior_incidence_mcmc = predictive_mcmc(
    rng_key=inference_process_mcmc.inference_prngkey,
    config=config_infer,  # arguments passed to `model`
    tf=200,
    obs_data=None,
    infer_mode=True,
)
assert inference_process_svi._inferer is not None, "mypy assert"
assert isinstance(inference_process_svi._inference_state, SVIRunResult)
predictive_svi = Predictive(
    model,
    guide=inference_process_svi._inferer.guide,
    params=inference_process_svi._inference_state.params,
    num_samples=1000,
)
posterior_incidence_svi = predictive_svi(
    rng_key=inference_process_mcmc.inference_prngkey,
    config=config_infer,  # arguments passed to `model`
    tf=200,
    obs_data=None,
    infer_mode=True,
)
print(posterior_incidence_mcmc.keys())
print(posterior_incidence_svi.keys())

# %%
# pick a random subset of 50 samples and plot the incidence, plot the true incidence from earlier as well
random_samples = jax.random.choice(
    inference_process_mcmc.inference_prngkey,
    posterior_incidence_mcmc["inf_incidence"].shape[0],
    shape=(50,),
)
for sample in random_samples:
    plt.plot(
        jnp.sum(posterior_incidence_mcmc["inf_incidence"][sample], axis=1),
        label=None,
    )
plt.plot(jnp.sum(incidence, axis=1), label="true incidence")
plt.legend()
plt.title("MCMC posterior predictive")
plt.show()

for sample in random_samples:
    plt.plot(
        jnp.sum(posterior_incidence_svi["inf_incidence"][sample], axis=1),
        label=None,
    )
plt.plot(jnp.sum(incidence, axis=1), label="true incidence")
plt.legend()
plt.title("SVI posterior predictive")
plt.show()
# %%
