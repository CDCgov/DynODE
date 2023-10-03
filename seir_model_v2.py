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
from config_base import ModelConfig as mc
from config_base import DataConfig as dc
from config_base import InferenceConfig as ic

# Use 4 cores
numpyro.set_host_device_count(4)
jax.config.update("jax_enable_x64", True)

#beta = S-E
#sigma = E-i
#gamma = i-R
def sir_ode(state, _, parameters):
    # Unpack state
    s, e, i, r, w1, w2, w3, w4 = state
    population = s + e + i + r + w1 + w2 + w3 + w4
    beta, sigma, gamma, contact_matrix, vax_rate, w1_protect, w2_protect, w3_protect, w4_protect, wanning_rate, mu = parameters
    ds_to_e = beta * s * contact_matrix.dot(i) / population #exposure
    dw1_to_e = beta * (1 - w1_protect) * w1 * contact_matrix.dot(i) / population # waning individuals exposure
    dw2_to_e = beta * (1 - w2_protect) * w2 * contact_matrix.dot(i) / population # waning individuals exposure
    dw3_to_e = beta * (1 - w3_protect) * w3 * contact_matrix.dot(i) / population # waning individuals exposure
    dw4_to_e = beta * (1 - w4_protect) * w4 * contact_matrix.dot(i) / population # waning individuals exposure

    ds_to_w1 = s * vax_rate #vaccination of suseptibles
    #we may want a dw1_to_w1 to represent recent infection getting vaccinated
    dw2_to_w1 = vax_rate * w2 #vaccination of previous immunity
    dw3_to_w1 = vax_rate * w3 #vaccination of previous immunity
    dw4_to_w1 = vax_rate * w4 #vaccination of previous immunity

    #TODO implement easy scaling of number of waning compartments

    dr_to_w1 = wanning_rate * r #waning
    dw1_to_w2 = wanning_rate * w1 #waning
    dw2_to_w3 = wanning_rate * w2 #waning
    dw3_to_w4 = wanning_rate * w3 #waning

    de_to_i = sigma * e #exposure -> infectious
    di_to_r = gamma * i #infectious -> recovered
    ds = s - ds_to_e - ds_to_w1
    de = e - de_to_i + ds_to_e + dw1_to_e + dw2_to_e + dw3_to_e + dw4_to_e 
    di = i + de_to_i - di_to_r
    dr = r + di_to_r - dr_to_w1
    dw1 = w1 + dr_to_w1 + ds_to_w1 + dw2_to_w1 + dw3_to_w1 + dw4_to_w1 - dw1_to_e - dw1_to_w2
    dw2 = w2 + dw1_to_w2 - dw2_to_e - dw2_to_w3
    dw3 = w3 + dw2_to_w3 - dw3_to_e - dw3_to_w4
    dw4 = w4 + dw3_to_w4 - dw4_to_e
    return (ds, de, di, dr, dw1, dw2, dw3, dw4)

rng = np.random.default_rng(seed=ic.MODEL_RAND_SEED)
target_population_fractions = rng.uniform(size = dc.NUM_AGE_GROUPS) #todo change so age distributions initialized non-uniformly
target_population_fractions = target_population_fractions / sum(target_population_fractions)
population = dc.POP_SIZE * dc.NUM_AGE_GROUPS * target_population_fractions 
population_fractions = population / sum(population)
# Normalize contact matrix to have unit spectral radius
#TODO move to trevors contact matrix shifted into age groups chosen.
contact_matrix = rng.uniform(size = (dc.NUM_AGE_GROUPS, dc.NUM_AGE_GROUPS))
contact_matrix = contact_matrix / max(abs(np.linalg.eigvals(contact_matrix)))

# Internal model parameters and state
initial_state = (population - mc.INITIAL_INFECTIONS * population_fractions, #s
                 mc.INITIAL_INFECTIONS * population_fractions, #e
                 population_fractions, #i
                 population_fractions, #r
                 population_fractions, #w1
                 population_fractions, #w2
                 population_fractions, #w3
                 population_fractions) #w4
beta = mc.SUBTYPE_SPECIFIC_R0[0] / mc.INFECTIOUS_PERIOD 
gamma = 1 / mc.INFECTIOUS_PERIOD 
sigma = 1 / mc.EXPOSED_TO_INFECTIOUS
wanning_rate = 1 / mc.WANING_1_TIME


solution = odeint(sir_ode, initial_state, jnp.linspace(0.0, 100.0, 101),
                  [beta, sigma, gamma, contact_matrix, mc.VACCINATION_RATE, mc.W1_PROTECT, mc.W2_PROTECT, mc.W3_PROTECT, mc.W4_PROTECT, wanning_rate, mc.BIRTH_RATE])
incidence = abs(-np.diff(solution[0], axis=0)) #TODO why did i need to add an abs() call here when there wasnt one before.
# Generate incidence sample
incidence_sample = rng.poisson(incidence)


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
    excess_r0 = numpyro.sample("excess_r0", dist.Exponential(1.0))
    r0 = numpyro.deterministic("r0", 1 + excess_r0)
        
    infectious_period = numpyro.sample("INCUBATION_RATE", dist.Exponential(1.0))
    w1_protect = numpyro.sample("w1_protect", dist.Exponential(1.0)) # protection against infection in first state of waning
    w2_protect = numpyro.sample("w2_protect", dist.Exponential(1.0)) # protection against infection in second state of waning
    w3_protect = numpyro.sample("w3_protect", dist.Exponential(1.0)) # protection against infection in third state of waning
    w4_protect = numpyro.sample("w4_protect", dist.Exponential(1.0)) # protection against infection in fourth state of waning
    recovered_to_waning = numpyro.sample("recovered_to_waning", dist.Normal(19.0)) #time in days before a recovered individual moves to first waned compartment
    birth_rate = mc.BIRTH_RATE
    vax_rate = mc.VACCINATION_RATE
    exposed_to_infectious = mc.EXPOSED_TO_INFECTIOUS

    # Transform parameters
    initial_state = (population - initial_infections * population_fractions, #s
                    initial_infections * population_fractions, #e
                    population_fractions, #i
                    population_fractions, #r
                    population_fractions, #sv
                    population_fractions, #ev
                    population_fractions, #iv
                    population_fractions) #w
    beta = r0 / infectious_period
    gamma = 1 / infectious_period
    sigma = 1 / exposed_to_infectious
    wanning_rate = 1 / recovered_to_waning
    mu = birth_rate

    solution = diffeqsolve(term, solver, t0, t1, dt0, initial_state,
                           args=[beta, sigma, gamma, contact_matrix, vax_rate, w1_protect, w2_protect, w3_protect, w4_protect, wanning_rate, mc.BIRTH_RATE], saveat=saveat)    

    model_incidence = -jnp.diff(solution.ys[0], axis=0)

    # Observed incidence
    numpyro.sample("incidence", dist.Poisson(model_incidence), obs=incidence)


# Perform inference
mcmc = MCMC(
    NUTS(model, dense_mass=True),
    num_warmup=ic.MCMC_NUM_WARMUP,
    num_samples=ic.MCMC_NUM_SAMPLES,
    num_chains=ic.MCMC_NUM_CHAINS,
    progress_bar=ic.MCMC_PROGRESS_BAR,
)
mcmc.run(PRNGKey(ic.MCMC_PRNGKEY),
         times=np.linspace(0.0, 100.0, 101),
         incidence=incidence)
mcmc.print_summary()