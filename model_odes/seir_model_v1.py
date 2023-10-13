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
from config.config_base import ModelConfig as mc
from config.config_base import DataConfig as dc
from config.config_base import InferenceConfig as ic

# Use 4 cores
numpyro.set_host_device_count(4)
jax.config.update("jax_enable_x64", True)


# beta = S-E
# sigma = E-i
# gamma = i-R
def sir_ode(state, _, parameters):
    # Unpack state
    s, e, i, r, sv, ev, iv, w = state
    population = s + e + i + r + sv + ev + iv + w
    (
        beta,
        sigma,
        gamma,
        contact_matrix,
        v_eff,
        nat_eff,
        wanning_rate,
        waned_rate,
        waned_vax,
        mu,
        vax_rate,
    ) = parameters
    dw_to_s = waned_rate * w
    dsv_to_s = waned_vax * sv
    ds_to_e = s * beta / population * contact_matrix.dot(i + iv)
    ds_to_sv = vax_rate * s
    dw_to_e = (1 - nat_eff) * w * beta / population * contact_matrix.dot(i + iv)
    de_to_i = sigma * e
    di_to_r = gamma * i
    dsv_to_ev = (1 - v_eff) * sv * beta / population * contact_matrix.dot(i + iv)
    dev_to_iv = sigma * ev
    div_to_r = gamma * iv
    dr_to_w = wanning_rate * r

    ds = (
        waned_rate * w
        + waned_vax * sv
        - s * beta / population * contact_matrix.dot(i + iv)
        - vax_rate * s
    )  # natural infected waned persons + vaccinated waned persons - suseptible * lambda * contacts - vaccinated individuals
    de = (
        s * beta / population * contact_matrix.dot(i + iv)
        + (1 - nat_eff) * w * beta / population * contact_matrix.dot(i + iv)
        - sigma * e
    )  # suseptible * lambda * contacts + waning_effectiveness * waned * lambda * contacts - exposed becoming infectious
    di = sigma * e - gamma * i  # exposed becoming infectious - individuals recovering

    dsv = (
        vax_rate * s
        - waned_vax * sv
        - (1 - v_eff) * sv * beta / population * contact_matrix.dot(i + iv)
    )  # new vaccinated - waned vaccination - (1-v_eff) * suseptible_vax * lambda * contacts
    dev = (1 - v_eff) * sv * beta / population * contact_matrix.dot(
        i + iv
    ) - sigma * ev  # (1-v_eff) * suseptible_vax * beta * contacts - exposed becoming infectious
    div = (
        sigma * ev - gamma * iv
    )  # exposed becoming infectious - individuals recovering

    dr = (
        gamma * i + gamma * iv - wanning_rate * r
    )  # recovering vax and non-vax individuals - waning individuals
    dw = (
        wanning_rate * r
        - waned_rate * w
        - (1 - nat_eff) * w * beta / population * contact_matrix.dot(i + iv)
    )  # waning individuals - fully waned back to suseptible - waning_effectiveness * waned * lambda * contacts
    # ds = ds + mu * population  # birth

    return (ds, de, di, dr, dsv, dev, div, dw)


rng = np.random.default_rng(seed=ic.MODEL_RAND_SEED)
target_population_fractions = np.random.uniform(
    size=dc.NUM_AGE_GROUPS
)  # todo change so age distributions initialized non-uniformly
target_population_fractions = target_population_fractions / sum(
    target_population_fractions
)
contact_matrix = np.random.rand(dc.NUM_AGE_GROUPS, dc.NUM_AGE_GROUPS)
population = dc.POP_SIZE * dc.NUM_AGE_GROUPS * target_population_fractions
population_fractions = population / sum(population)
# Normalize contact matrix to have unit spectral radius
contact_matrix = contact_matrix / max(abs(np.linalg.eigvals(contact_matrix)))

# Internal model parameters and state
inital_suseptible = population
initial_state = (
    population - mc.INITIAL_INFECTIONS * population_fractions,  # s
    mc.INITIAL_INFECTIONS * population_fractions,  # e
    population_fractions,  # i
    population_fractions,  # r
    population_fractions,  # sv
    population_fractions,  # ev
    population_fractions,  # iv
    population_fractions,
)  # w
beta = mc.SUBTYPE_SPECIFIC_R0[0] / mc.INFECTIOUS_PERIOD
gamma = 1 / mc.INFECTIOUS_PERIOD
sigma = 1 / mc.EXPOSED_TO_INFECTIOUS
wanning_rate = 1 / mc.WANING_1_TIME
waned_rate = 1 / mc.WANED_TO_SUSEPTIBLE


# Solve ODE old solution
solution = odeint(
    sir_ode,
    initial_state,
    jnp.linspace(0.0, 100.0, 101),
    [
        beta,
        sigma,
        gamma,
        contact_matrix,
        mc.VACCINE_EFFECTIVENESS,
        mc.NAT_IMMUNE_EFFECTIVENESS,
        wanning_rate,
        waned_rate,
        mc.VACCINE_WANING,
        mc.BIRTH_RATE,
        mc.VACCINATION_RATE,
    ],
)
incidence = abs(
    -np.diff(solution[0], axis=0)
)  # TODO why did i need to add an abs() call here when there wasnt one before.
# Generate incidence sample
rng = np.random.default_rng(seed=ic.MODEL_RAND_SEED)
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
    initial_infections = numpyro.sample("initial_infections", dist.Exponential(1.0))
    excess_r0 = numpyro.sample("excess_r0", dist.Exponential(1.0))
    r0 = numpyro.deterministic("r0", 1 + excess_r0)

    infectious_period = numpyro.sample("INCUBATION_RATE", dist.Exponential(1.0))
    vaccine_eff = numpyro.sample(
        "vaccine_eff", dist.Exponential(1.0)
    )  # here for old model
    nat_immune_eff = numpyro.sample(
        "nat_immune_eff", dist.Exponential(1.0)
    )  # % effectiveness of prior natural immunity in waned state at preventing infection
    recovered_to_waning = numpyro.sample(
        "recovered_to_waning", dist.Normal(19.0)
    )  # time in days before a recovered individual moves to first waned compartment
    waning_to_suseptible = numpyro.sample("waning_to_suseptible", dist.Normal(19.0))  #
    vaccine_waning = numpyro.sample("vaccine_waning", dist.Normal(24.0))
    birth_rate = mc.BIRTH_RATE
    vax_rate = mc.VACCINATION_RATE
    exposed_to_infectious = mc.EXPOSED_TO_INFECTIOUS

    # Transform parameters
    initial_state = (
        population - initial_infections * population_fractions,  # s
        initial_infections * population_fractions,  # e
        population_fractions,  # i
        population_fractions,  # r
        population_fractions,  # sv
        population_fractions,  # ev
        population_fractions,  # iv
        population_fractions,
    )  # w
    beta = r0 / infectious_period
    gamma = 1 / infectious_period
    sigma = 1 / exposed_to_infectious
    wanning_rate = 1 / recovered_to_waning
    waned_vax = 1 / waning_to_suseptible
    mu = birth_rate

    solution = diffeqsolve(
        term,
        solver,
        t0,
        t1,
        dt0,
        initial_state,
        args=(
            beta,
            sigma,
            gamma,
            contact_matrix,
            vaccine_eff,
            nat_immune_eff,
            wanning_rate,
            waned_rate,
            waned_vax,
            mu,
            vax_rate,
        ),
        saveat=saveat,
    )
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
mcmc.run(
    PRNGKey(ic.MCMC_PRNGKEY), times=np.linspace(0.0, 100.0, 101), incidence=incidence
)
mcmc.print_summary()
