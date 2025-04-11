# An Example of how to simulate a basic SIR compartmental model using the dynode package
# Including all the class setup
# %% imports and definitions
# most of these imports are for type hinting
import arviz as az
import chex
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpyro
from diffrax import Solution
from numpyro.infer import Predictive

from dynode.model_configuration import (
    SimulationConfig,
    TransmissionParams,
)
from dynode.model_configuration.inference import MCMCProcess, SVIProcess
from dynode.model_configuration.pre_packaged.example_sir_config import (
    SIRConfig,
    SIRInferedConfig,
)
from dynode.odes import AbstractODEParams, simulate
from dynode.sample import sample_then_resolve
from dynode.typing import CompartmentGradients, CompartmentState


# define the behavior of the ODEs and the parameters they take
@chex.dataclass
class SIR_ODEParams(AbstractODEParams):
    beta: chex.ArrayDevice  # r0/infectious period
    gamma: chex.ArrayDevice  # 1/infectious period
    pass


# define a function to easily translate the object oriented TransmissionParams
# into the vectorized ODEParams.
def get_odeparams(transmission_params: TransmissionParams) -> SIR_ODEParams:
    """Transform and vectorize transmission parameters into ODE parameters."""
    transmission_params = sample_then_resolve(transmission_params)
    strain = transmission_params.strains[0]
    beta = strain.r0 / strain.infectious_period
    gamma = 1 / strain.infectious_period
    return SIR_ODEParams(beta=jnp.array(beta), gamma=jnp.array(gamma))


# define your Jit compiled ODE function
@jax.jit
def sir_ode(
    t: float, state: CompartmentState, p: SIR_ODEParams
) -> CompartmentGradients:
    """A simple SIR ODE model with no time-varying components."""
    s, i, r = state
    pop_size = s + i + r
    s_to_i = (p.beta * s * i) / pop_size
    i_to_r = i * p.gamma
    ds = -s_to_i
    di = s_to_i - i_to_r
    dr = i_to_r
    return tuple([ds, di, dr])


# %% setup simulation process.
# instantiate the config
config_static = SIRConfig()


def run_simulation(config: SimulationConfig, tf) -> Solution:
    ode_params = get_odeparams(config.parameters.transmission_params)

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
    obs_data: jax.Array = None,
    infer_mode=False,
):
    """Numpyro model for simulating infection incidence of an SIR model."""
    solution = run_simulation(config, tf)
    # compare to observed data if we have it
    if infer_mode:
        incidence = jnp.diff(solution.ys[2].flatten())
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
plt.plot(solution.ys[0], label="s")
plt.plot(solution.ys[1], label="i")
plt.plot(solution.ys[2], label="r")
plt.legend()
plt.show()
# diff recovered individuals to recover lagged incidence.
incidence = jnp.diff(solution.ys[2].flatten())

# %%
# set up inference process
# now lets infer the parameters of this strain instead
config_infer = SIRInferedConfig()
# creating two InferenceProcesses, one for MCMC and one for SVI
inference_process_mcmc = MCMCProcess(
    simulator=model,
    num_warmup=1000,
    num_samples=1000,
    num_chains=1,
    nuts_max_tree_depth=10,
)
inference_process_svi = SVIProcess(
    simulator=model,
    num_iterations=2000,
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
    f"True value or R0: {config_static.parameters.transmission_params.strains[0].r0} "
    f"Infectious Period: {config_static.parameters.transmission_params.strains[0].infectious_period}"
)
# notice the name of the posterior sample mimics the index of `transmission_params.strains`
# this will help you find parameters later on.
print(
    f"MCMC posteriors R0: {jnp.mean(posterior_samples_mcmc['strains_0_r0'])}, "
    f"Infectious Period: {jnp.mean(posterior_samples_mcmc['strains_0_infectious_period'])}"
)
print(
    f"SVI posteriors R0: {jnp.mean(posterior_samples_svi['strains_0_r0'])}, "
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
fig.suptitle("Density Intervals for R0")

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
# pick a random subset of 50 samples and plot the incidence, plot the true incidence from earilier as well
random_samples = jax.random.choice(
    inference_process_mcmc.inference_prngkey,
    posterior_incidence_mcmc["inf_incidence"].shape[0],
    shape=(50,),
)
for sample in random_samples:
    plt.plot(posterior_incidence_mcmc["inf_incidence"][sample], label=None)
plt.plot(incidence, label="true incidence")
plt.legend()
plt.title("MCMC posterior predictive")
plt.show()

for sample in random_samples:
    plt.plot(posterior_incidence_svi["inf_incidence"][sample], label=None)
plt.plot(incidence, label="true incidence")
plt.legend()
plt.title("SVI posterior predictive")
plt.show()
# %%
