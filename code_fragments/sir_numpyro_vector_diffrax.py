from jax.experimental.ode import odeint
import jax.numpy as jnp
import numpy as np
from jax.random import PRNGKey
import jax.config

import numpyro
import numpyro.distributions as dist
from numpyro.distributions.transforms import AffineTransform

from numpyro.infer import MCMC, NUTS, Predictive
from numpyro.infer.reparam import TransformReparam

from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5

# Use 4 cores
numpyro.set_host_device_count(4)
jax.config.update("jax_enable_x64", True)

def sir_ode(state, _, parameters):
    # Unpack state
    s, i, r = state
    beta, gamma, contact_matrix = parameters
    population = s + i + r

    # Compute flows 
    ds_to_i = beta * s / population * contact_matrix.dot(i)
    di_to_r = gamma * i

    # Compute derivatives
    ds = -ds_to_i
    di = ds_to_i - di_to_r
    dr = di_to_r

    return (ds, di, dr) #jnp.stack([ds, di, dr])


rng = np.random.default_rng(seed=867530)
n_age_groups = 50
target_population_fractions = np.random.uniform(size = n_age_groups)
target_population_fractions = target_population_fractions / sum(target_population_fractions)
contact_matrix = np.random.rand(n_age_groups, n_age_groups)

# External parameters
r0 = 1.5
infectious_period = 3.0
population = 10000 * n_age_groups * target_population_fractions #np.array([.25, .75])
population_fractions = population / sum(population)
initial_infections = 10.0
# contact_matrix = np.array([[18,  3],
# 					       [ 9, 12]])
# Normalize contact matrix to have unit spectral radius
contact_matrix = contact_matrix / max(abs(np.linalg.eigvals(contact_matrix)))

# Internal model parameters and state
initial_state = (population - initial_infections * population_fractions, #s
                 initial_infections * population_fractions, #i
                 population_fractions) #r
beta = r0 / infectious_period
gamma = 1 / infectious_period


# Solve ODE
solution = odeint(sir_ode, initial_state, jnp.linspace(0.0, 100.0, 101),
                  [beta, gamma, contact_matrix])
incidence = -np.diff(solution[0], axis=0)

# Generate incidence sample
rng = np.random.default_rng(seed=8675309)
incidence_sample = rng.poisson(incidence)

print(incidence_sample)

# Prep diffrax if needed
term = ODETerm(lambda t, state, parameters: sir_ode(state, t, parameters))
solver = Tsit5()
t0 = 0.0
t1 = 100.0
dt0 = 0.1
saveat = SaveAt(ts=jnp.linspace(t0, t1, 101))

from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5

def model(times, incidence):
    # Parameters
    initial_infections = numpyro.sample("initial_infections",
                                        dist.Exponential(1.0))
    
    # reparam_config = {"r0": TransformReparam()}
    # with numpyro.handlers.reparam(config=reparam_config):
    #     r0 = numpyro.sample("r0",
    #                         dist.TransformedDistribution(
    #                             dist.Exponential(1.0),
    #                             AffineTransform(1.0, 1.0)))
    excess_r0 = numpyro.sample("excess_r0", dist.Exponential(1.0))
    r0 = numpyro.deterministic("r0", 1 + excess_r0)
        
    infectious_period = numpyro.sample("infectious_period", dist.Exponential(1.0))

    # Transform parameters
    initial_state = (population - initial_infections * population_fractions, #s
                     initial_infections * population_fractions, #i
                     population_fractions) #r
    beta = r0 / infectious_period
    gamma = 1 / infectious_period

    # Integrate the model
    # solution = odeint(sir_ode, initial_state, times,
    #                   [beta, gamma, contact_matrix])
    # model_incidence = -jnp.diff(solution[0], axis=0)
    solution = diffeqsolve(term, solver, t0, t1, dt0, initial_state,
                           args=(beta, gamma, contact_matrix), saveat=saveat)
    model_incidence = -jnp.diff(solution.ys[0], axis=0)

    # Observed incidence
    numpyro.sample("incidence", dist.Poisson(model_incidence), obs=incidence)


# Perform inference
mcmc = MCMC(
    NUTS(model, dense_mass=True),
    num_warmup=1000,
    num_samples=1000,
    num_chains=4,
    progress_bar=True,
)
mcmc.run(PRNGKey(8675309),
         times=np.linspace(0.0, 100.0, 101),
         incidence=incidence)
mcmc.print_summary()