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
from config_base import ModelConfig as mc
from config_base import DataConfig as dc
from config_base import InferenceConfig as ic
import utils


class BasicMechanisticModel:
    "Implementation of a basic Mechanistic model for scenario analysis"

    def __init__(
        self,
        R0_dist=mc.STRAIN_SPECIFIC_R0,
        waning_protect_dist=mc.WANING_PROTECTIONS,
        init_pop_size=dc.POP_SIZE,
        target_population_fractions=None,
        contact_matrix=None,
        init_infection_dist=None,
    ):
        rng = np.random.default_rng(seed=ic.MODEL_RAND_SEED)
        self.R0_dist = R0_dist
        self.waning_protect_dist = waning_protect_dist
        # if not given, uniformally generate population fractions.
        if not target_population_fractions:
            target_population_fractions = rng.uniform(
                size=dc.NUM_AGE_GROUPS
            )  # TODO change so age distributions initialized non-uniformly
            target_population_fractions = target_population_fractions / sum(
                target_population_fractions
            )
        self.target_population_fractions = target_population_fractions
        self.population = init_pop_size * self.target_population_fractions

        # if not given, load contact matrices via Dina's mixing data.
        if not contact_matrix:
            contact_matricies = utils.load_demographic_data()
            contact_matrix = contact_matricies["United States"]["oth_CM"]
        self.contact_matrix = contact_matrix

        # if not given an inital infection distribution, use max eig value vector
        if not init_infection_dist:
            eig_data = np.linalg.eig(contact_matrix)
            max_index = np.argmax(eig_data[0])
            init_infection_dist = eig_data[1][:, max_index]

        # with inital infection distribution by age group, break down uniformally by number of strains.
        initial_infections_by_strain = (
            mc.INITIAL_INFECTIONS
            * init_infection_dist[:, None]
            * np.ones(dc.NUM_STRAINS)
            / dc.NUM_STRAINS
        )
        self.init_infection_dist = init_infection_dist
        self.initial_state = (
            self.population - mc.INITIAL_INFECTIONS * self.init_infection_dist,  # s
            initial_infections_by_strain,  # e
            np.zeros((dc.NUM_AGE_GROUPS, dc.NUM_STRAINS)),  # i
            np.zeros((dc.NUM_AGE_GROUPS, dc.NUM_STRAINS)),  # r
            np.zeros((dc.NUM_AGE_GROUPS, dc.NUM_STRAINS, mc.NUM_WANING_COMPARTMENTS)),
        )  # w

    def sample_r0(self):
        """sample r0 for each strain according to an exponential distribution
        with rate equal to strain specific r0 in config file"""
        r0s = []
        for i, sample_mean in enumerate(self.R0_dist):
            excess_r0 = numpyro.sample(
                "excess_r0_" + str(i), dist.Exponential(sample_mean)
            )
            r0 = numpyro.deterministic("r0_" + str(i), 1 + excess_r0)
            r0s.append(r0)
        return r0s

    def sample_waning_protections(self):
        """Sample a waning rate for each of the waning comparments according to an exponential distribution
        with rate equal to config file rates."""
        waning_rates = []
        for i, sample_mean in enumerate(self.waning_protect_dist):
            excess_r0 = numpyro.sample(
                "waning_rates_" + str(i), dist.Exponential(sample_mean)
            )
            waning_rate = numpyro.deterministic("waning_rates_" + str(i), 1 + excess_r0)
            waning_rates.append(waning_rate)

    def incidence(self, model, incidence):
        term = ODETerm(lambda t, state, parameters: model(state, t, parameters))
        solver = Tsit5()
        t0 = 0.0
        t1 = 100.0
        dt0 = 0.1
        saveat = SaveAt(ts=jnp.linspace(t0, t1, 101))

        r0 = self.sample_r0()
        waning_protections = self.sample_waning_protections()
        beta = r0 / mc.INFECTIOUS_PERIOD
        gamma = 1 / mc.INFECTIOUS_PERIOD
        sigma = 1 / mc.EXPOSED_TO_INFECTIOUS
        wanning_rate = 1 / mc.WANING_TIME
        suseptibility_matrix = np.ones((dc.NUM_STRAINS, dc.NUM_STRAINS))

        solution = diffeqsolve(
            term,
            solver,
            t0,
            t1,
            dt0,
            self.initial_state,
            args=[
                beta,
                sigma,
                gamma,
                self.contact_matrix,
                mc.VACCINATION_RATE,
                waning_protections,
                wanning_rate,
                mc.BIRTH_RATE,
                suseptibility_matrix,
            ],
            saveat=saveat,
        )

        model_incidence = -jnp.diff(solution.ys[0], axis=0)

        # Observed incidence
        numpyro.sample("incidence", dist.Poisson(model_incidence), obs=incidence)

    def infer(self, model, incidence):
        mcmc = MCMC(
            NUTS(model, dense_mass=True),
            num_warmup=ic.MCMC_NUM_WARMUP,
            num_samples=ic.MCMC_NUM_SAMPLES,
            num_chains=ic.MCMC_NUM_CHAINS,
            progress_bar=ic.MCMC_PROGRESS_BAR,
        )
        mcmc.run(
            PRNGKey(ic.MCMC_PRNGKEY),
            times=np.linspace(0.0, 100.0, 101),
            incidence=incidence,
        )
        mcmc.print_summary()

    def run(self, model, tf=100.0, save=True, save_path="model_run.png"):
        """
        runs the mechanistic model using beta, gamma, sigma, and waning rate based on config file values.
        Does not sample waning protections or R0 by strain.
        """
        term = ODETerm(lambda t, state, parameters: model(state, t, parameters))
        solver = Tsit5()
        t0 = 0.0
        dt0 = 0.1
        saveat = SaveAt(ts=jnp.linspace(t0, tf, int(tf) + 1))

        beta = mc.STRAIN_SPECIFIC_R0 / mc.INFECTIOUS_PERIOD
        gamma = 1 / mc.INFECTIOUS_PERIOD
        sigma = 1 / mc.EXPOSED_TO_INFECTIOUS
        wanning_rate = 1 / mc.WANING_TIME
        solution = diffeqsolve(
            term,
            solver,
            t0,
            tf,
            dt0,
            self.initial_state,
            args=[
                beta,
                sigma,
                gamma,
                self.contact_matrix,
                mc.VACCINATION_RATE,
                mc.WANING_PROTECTIONS,
                wanning_rate,
                mc.BIRTH_RATE,
                self.population,
                jnp.ones((dc.NUM_STRAINS, dc.NUM_STRAINS)),
            ],
            saveat=saveat,
            max_steps=30000,
        )
        save_path = (
            save_path if save else None
        )  # dont set a save path if we dont want to save
        fig, ax = utils.plot_diffrax_solution(
            solution.ys,
            plot_compartments=["s", "e", "i", "r", "w0", "w1", "w2", "w3"],
            save_path=save_path,
        )
        return solution
