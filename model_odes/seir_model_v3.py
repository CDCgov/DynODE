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

numpyro.set_host_device_count(4)
jax.config.update("jax_enable_x64", True)
from config.config_base import ModelConfig as mc
from config.config_base import DataConfig as dc
from config.config_base import InferenceConfig as ic
import utils

# Use 4 cores


# beta = S-E
# sigma = E-i
# gamma = i-R
def sir_ode(state, _, parameters):
    # Unpack state
    # dims s = (dc.NUM_AGE_GROUPS)
    # dims e/i/r = (dc.NUM_AGE_GROUPS, dc.NUM_STRAINS)
    # dims w = (dc.NUM_AGE_GROUPS, dc.NUM_STRAINS, mc.NUM_WANING_COMPARTMENTS)
    s, e, i, r, w = state
    (
        beta,
        sigma,
        gamma,
        contact_matrix,
        vax_rate,
        waning_protections,
        wanning_rate,
        mu,
        population,
        susceptibility_matrix,
    ) = parameters

    # TODO when adding birth and deaths just create it as a compartment
    # contact_matrix.dot(i) = 5x5.dot(5x3)
    # TODO make beta 1x3 vector of beta by strain
    force_of_infection = beta * contact_matrix.dot(i) / population[:, None]
    # force of infection = 5x3 (dc.NUM_AGE_GROUPS, dc.NUM_STRAINS)
    ds_to_e = force_of_infection * s[:, None]

    # TODO what do we do about strain here? what strain are vaccinated people placed into
    ds_to_w1 = s * vax_rate  # vaccination of suseptibles
    # we may want a dw1_to_w1 to represent recent infection getting vaccinated

    de_to_i = sigma * e  # exposure -> infectious
    di_to_r = gamma * i  # infectious -> recovered
    dr_to_w = wanning_rate * r
    # guaranteed to wane into first waning compartment remaining in their strains.

    dw = np.zeros(w.shape)
    dw_to_e_arr = np.zeros(
        (dc.NUM_AGE_GROUPS, dc.NUM_STRAINS, mc.NUM_WANING_COMPARTMENTS)
    )
    for strain_source_idx in range(dc.NUM_STRAINS):
        force_of_infection_strain = force_of_infection[:, strain_source_idx]
        for strain_target_idx in range(dc.NUM_STRAINS):
            # strain 1 will attempt to infect those previously infected with strain 2.
            ws_by_age = w[:, strain_target_idx, :]
            # (dc.NUM_AGE_GROUPS, mc.NUM_WANING_COMPARTMENTS)
            partial_susceptibility = susceptibility_matrix[
                strain_source_idx, strain_target_idx
            ]
            #            (dc.NUM_AGE_GROUPS) * (dc.NUM_AGE_GROUPS, mc.NUM_WANING_COMPARTMENTS).dot((1 - mc.NUM_WANING_COMPARTMENTS))
            ws_exposed = force_of_infection_strain * ws_by_age.dot(
                (1 - waning_protections) * partial_susceptibility
            )  # waning individuals exposure
            dw[:, strain_target_idx, :] -= ws_exposed
            de[:, strain_source_idx] += ws_exposed
            dw_to_e_arr[:, strain_source_idx] += ws_exposed
            # TODO check this, moving from waning type2 -> exposed type1
    # only operations that have no competition element may be done below
    for w_idx in mc.w_idx:  # loop through the waning comparments:
        w_waned = (
            0
            if w_idx == mc.NUM_WANING_COMPARTMENTS - 1
            else wanning_rate * w[:, :, w_idx]
        )
        # last compartment doesnt wane
        r_to_w = dr_to_w if w_idx == 0 else 0
        # only top waning compartment receives people from "r"
        w_vax = 0
        if w_idx == 0:
            # other waned compartments can get vaccinated into first waned compartment
            w_vax = sum(
                [
                    vax_rate * w[:, :, w_idx_loop]
                    for w_idx_loop in mc.w_idx
                    if w_idx_loop != 0
                ]
            )
            w_vax += (
                ds_to_w1  # suseptibles also get vaccinated, TODO is this shape right?
            )
        else:  # vaccination takes away from non-first compartments.
            w_vax = -vax_rate * w[:, :, w_idx]
    ds = (
        np.zeros(s.shape) - np.sum(ds_to_e, axis=1) - ds_to_w1
    )  # sum ds_to_e since s does not split by subtype
    de = (
        np.zeros(e.shape) - de_to_i + ds_to_e + sum(dw_to_e_arr, axis=2)
    )  # sum here since all waning compartments go to e.
    di = np.zeros(i.shape) + de_to_i - di_to_r
    dr = np.zeros(r.shape) + di_to_r - dr_to_w

    # ds =  - ds_to_e - ds_to_w1
    # de =  - de_to_i + ds_to_e + sum(dw_to_e_arr)
    # di =  + de_to_i - di_to_r
    # dr =  + di_to_r - dr_to_w1
    # dw1 =  + dr_to_w1 + ds_to_w1 + dw2_to_w1 + dw3_to_w1 + dw4_to_w1 - dw1_to_e - dw1_to_w2 #TODO how to fix these?
    # dw2 =  + dw1_to_w2 - dw2_to_e - dw2_to_w3 - dw2_to_w1
    # dw3 =  + dw2_to_w3 - dw3_to_e - dw3_to_w4 - dw3_to_w1
    # dw4 =  + dw3_to_w4 - dw4_to_e - dw4_to_w1
    return (ds, de, di, dr, dw)


rng = np.random.default_rng(seed=ic.MODEL_RAND_SEED)
target_population_fractions = rng.uniform(
    size=dc.NUM_AGE_GROUPS
)  # todo change so age distributions initialized non-uniformly
target_population_fractions = target_population_fractions / sum(
    target_population_fractions
)
population = dc.POP_SIZE * dc.NUM_AGE_GROUPS * target_population_fractions
population_fractions = population / sum(population)
# Normalize contact matrix to have unit spectral radius
# TODO move to trevors contact matrix shifted into age groups chosen.
# contact_matrix = rng.uniform(size = (dc.NUM_AGE_GROUPS, dc.NUM_AGE_GROUPS))
# contact_matrix = contact_matrix / max(abs(np.linalg.eigvals(contact_matrix)))
contact_matrices = utils.load_demographic_data()
contact_matrix = contact_matrices["United States"]["oth_CM"]
eig_data = np.linalg.eig(contact_matrix)
max_index = np.argmax(eig_data[0])
initial_infection_distribution = eig_data[1][:, max_index]
initial_infections_by_strain = (
    mc.INITIAL_INFECTIONS
    * initial_infection_distribution[:, None]
    * np.ones(dc.NUM_STRAINS)
    / dc.NUM_STRAINS
)


# Internal model parameters and state
# initial_state = (population - mc.INITIAL_INFECTIONS * population_fractions, #s
#                  mc.INITIAL_INFECTIONS * population_fractions, #e
#                  population_fractions, #i
#                  population_fractions, #r
#                  population_fractions, #w1
#                  population_fractions, #w2
#                  population_fractions, #w3
#                  population_fractions) #w4

initial_state = (
    population - mc.INITIAL_INFECTIONS * initial_infection_distribution,  # s
    initial_infections_by_strain,  # e
    np.zeros((dc.NUM_AGE_GROUPS, dc.NUM_STRAINS)),  # i
    np.zeros((dc.NUM_AGE_GROUPS, dc.NUM_STRAINS)),  # r
    np.zeros((dc.NUM_AGE_GROUPS, dc.NUM_STRAINS, mc.NUM_WANING_COMPARTMENTS)),
)  # w
beta = mc.STRAIN_SPECIFIC_R0[0] / mc.INFECTIOUS_PERIOD
gamma = 1 / mc.INFECTIOUS_PERIOD
sigma = 1 / mc.EXPOSED_TO_INFECTIOUS
wanning_rate = 1 / mc.WANING_TIME


solution = odeint(
    sir_ode,
    initial_state,
    jnp.linspace(0.0, 100.0, 101),
    [
        beta,
        sigma,
        gamma,
        contact_matrix,
        mc.VACCINATION_RATE,
        mc.WANING_PROTECTIONS,
        wanning_rate,
        mc.BIRTH_RATE,
        population,
        np.ones((dc.NUM_STRAINS, dc.NUM_STRAINS)),
    ],
)
fig, ax = utils.plot_ode_solution(
    solution,
    plot_compartments=["s", "e", "i", "r", "w1", "w2", "w3", "w4"],
    save_path="testing_image_no_vaxs.png",
)

incidence = abs(
    -np.diff(solution[0], axis=0)
)  # TODO why did i need to add an abs() call here when there wasnt one before.
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
    initial_infections = numpyro.sample("initial_infections", dist.Exponential(1.0))
    excess_r0 = numpyro.sample("excess_r0", dist.Exponential(1.0))
    r0 = numpyro.deterministic("r0", 1 + excess_r0)

    infectious_period = numpyro.sample("INCUBATION_RATE", dist.Exponential(1.0))
    w1_protect = numpyro.sample(
        "w1_protect", dist.Exponential(1.0)
    )  # protection against infection in first state of waning
    w2_protect = numpyro.sample(
        "w2_protect", dist.Exponential(1.0)
    )  # protection against infection in second state of waning
    w3_protect = numpyro.sample(
        "w3_protect", dist.Exponential(1.0)
    )  # protection against infection in third state of waning
    w4_protect = numpyro.sample(
        "w4_protect", dist.Exponential(1.0)
    )  # protection against infection in fourth state of waning
    recovered_to_waning = numpyro.sample(
        "recovered_to_waning", dist.Normal(19.0)
    )  # time in days before a recovered individual moves to first waned compartment
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
    mu = birth_rate

    solution = diffeqsolve(
        term,
        solver,
        t0,
        t1,
        dt0,
        initial_state,
        args=[
            beta,
            sigma,
            gamma,
            contact_matrix,
            vax_rate,
            w1_protect,
            w2_protect,
            w3_protect,
            w4_protect,
            wanning_rate,
            mc.BIRTH_RATE,
            np.ones((dc.NUM_STRAINS, dc.NUM_STRAINS)),
        ],
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
