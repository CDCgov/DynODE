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
import config.config_base as config_base
from config.config_base import ModelConfig as mc
from config.config_base import DataConfig as dc
from config.config_base import InferenceConfig as ic
import utils

# plotting libraries
import matplotlib.pyplot as plt


class BasicMechanisticModel:
    "Implementation of a basic Mechanistic model for scenario analysis"

    def __init__(
        self,
        num_age_groups=mc.NUM_AGE_GROUPS,
        num_strains=mc.NUM_STRAINS,
        age_limits=mc.AGE_LIMITS,
        init_pop_size=mc.POP_SIZE,
        birth_rate=mc.BIRTH_RATE,
        infectious_period=mc.INFECTIOUS_PERIOD,
        exposed_to_infectious=mc.EXPOSED_TO_INFECTIOUS,
        vaccination_rate=mc.VACCINATION_RATE,
        initial_infections=mc.INITIAL_INFECTIONS,
        R0_dist=mc.STRAIN_SPECIFIC_R0,
        num_waning_compartments=mc.NUM_WANING_COMPARTMENTS,
        waning_protect_dist=mc.WANING_PROTECTIONS,
        waning_time=mc.WANING_TIME,
        num_compartments=mc.NUM_COMPARTMENTS,
        w_idx=mc.w_idx,
        idx=mc.idx,
        target_population_fractions=None,
        contact_matrix=None,
        init_infection_dist=None,
        waning_distribution=None,
    ):
        self.num_age_groups = num_age_groups
        self.num_strains = num_strains
        self.age_limits = age_limits
        self.init_pop_size = init_pop_size
        self.birth_rate = birth_rate
        self.infectious_period = infectious_period
        self.exposed_to_infectious = exposed_to_infectious
        self.vaccination_rate = vaccination_rate
        self.initial_infections = initial_infections
        self.R0_dist = R0_dist
        self.num_waning_compartments = num_waning_compartments
        self.waning_protect_dist = waning_protect_dist
        self.waning_time = waning_time
        self.num_compartments = num_compartments
        self.w_idx = w_idx
        self.idx = idx
        rng = np.random.default_rng(seed=ic.MODEL_RAND_SEED)
        # if not given, generate population fractions based on observed census data
        if not target_population_fractions:
            target_population_fractions = utils.load_age_demographics()["United States"]

        self.target_population_fractions = target_population_fractions
        self.population = init_pop_size * self.target_population_fractions

        # if not given, load contact matrices via Dina's mixing data.
        if not contact_matrix:
            contact_matricies = utils.load_demographic_data()
            contact_matrix = contact_matricies["United States"]["oth_CM"]
        self.contact_matrix = contact_matrix

        if not waning_distribution:
            path = "data/serological-data/Nationwide_Commercial_Laboratory_Seroprevalence_Survey_20231018.csv"
            waning_distribution = utils.load_serology_demographics(
                path,
                self.age_limits,
                self.waning_time,
                self.num_waning_compartments,
                self.num_strains,
            )

        # if not given an inital infection distribution, use max eig value vector
        if not init_infection_dist:
            eig_data = np.linalg.eig(contact_matrix)
            max_index = np.argmax(eig_data[0])
            init_infection_dist = abs(eig_data[1][:, max_index])
            init_infection_dist = init_infection_dist / sum(init_infection_dist)

        # with inital infection distribution by age group, break down uniformally by number of strains.
        # TODO non-uniform strain distribution if needed.
        initial_infections_by_strain = (
            initial_infections
            * init_infection_dist[:, None]
            * np.ones(num_strains)
            / num_strains
        )
        self.init_infection_dist = init_infection_dist
        self.initial_state = (
            self.population - self.initial_infections * self.init_infection_dist,  # s
            np.zeros((num_age_groups, num_strains)),  # e
            initial_infections_by_strain,  # i
            np.zeros((num_age_groups, num_strains)),  # r
            np.zeros((num_age_groups, num_strains, num_waning_compartments)),
        )  # w

    def get_args(self, sample=False):
        if sample:
            beta = utils.sample_r0(self.R0_dist) / self.infectious_period
            waning_protections = utils.sample_waning_protections(
                self.waning_protect_dist
            )
        else:  # if we arent sampling we use values from config in self
            beta = self.R0_dist / self.infectious_period
            waning_protections = self.waning_protect_dist

        gamma = 1 / self.infectious_period
        sigma = 1 / self.exposed_to_infectious
        wanning_rate = 1 / self.waning_time
        suseptibility_matrix = jnp.ones(
            (self.num_strains, self.num_strains)
        )  # TODO use priors here
        args = [  # TODO convert to dictionary if diffeqsolve() allows for args to be a dict.
            beta,
            sigma,
            gamma,
            self.contact_matrix,
            self.vaccination_rate,
            waning_protections,
            wanning_rate,
            self.birth_rate,
            self.population,
            suseptibility_matrix,
            self.num_strains,
            self.num_waning_compartments,
        ]
        return args

    def incidence(self, model, incidence):
        term = ODETerm(lambda t, state, parameters: model(state, t, parameters))
        solver = Tsit5()
        t0 = 0.0
        t1 = 100.0
        dt0 = 0.1
        saveat = SaveAt(ts=jnp.linspace(t0, t1, 101))
        solution = diffeqsolve(
            term,
            solver,
            t0,
            t1,
            dt0,
            self.initial_state,
            args=self.get_args(sample=True),
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
        solution = diffeqsolve(
            term,
            solver,
            t0,
            tf,
            dt0,
            self.initial_state,
            args=self.get_args(sample=False),
            saveat=saveat,
            max_steps=30000,
        )
        save_path = (
            save_path if save else None
        )  # dont set a save path if we dont want to save
        fig, ax = self.plot_diffrax_solution(
            solution,
            plot_compartments=["s", "e", "i", "r", "w0", "w1", "w2", "w3"],
            save_path=save_path,
        )
        return solution

    def plot_diffrax_solution(
        self,
        sol,
        plot_compartments=["s", "e", "i", "r"],
        save_path=None,
    ):
        """
        plots a run from diffeqsolve() with compartments `plot_compartments` returning figure and axis.
        If `save_path` is not None will save figure to that path attached with meta data in `meta_data`.

        Parameters
        ----------
        sol : difrax.Solution
            object containing ODE run
        plot_compartment : list(str)
            compartment titles as defined by the config file used to initialize self
        save_path : str
            if `save_path = None` do not save figure to output directory. Otherwise save to relative path `save_path`
            attaching meta data of the self object.
        """
        sol = sol.ys
        get_indexes = []
        for compartment in plot_compartments:
            if "W" in compartment.upper():
                # waning compartments are held in a different manner, we need two indexes to access them
                index_slice = [
                    self.idx.__getitem__("W"),
                    self.w_idx.__getitem__(compartment.strip().upper()),
                ]
                get_indexes.append(index_slice)
            else:
                get_indexes.append(self.idx.__getitem__(compartment.strip().upper()))

        fig, ax = plt.subplots(1)
        for compartment, idx in zip(plot_compartments, get_indexes):
            if "W" in compartment.upper():  # then idx=(idx.W, w_idx.W1/2/....)
                # if we are plotting a waning compartment, we need to parse 1 extra dimension
                sol_compartment = np.array(sol[idx[0]])[:, :, :, idx[1]]
            else:
                # non-waning compartments dont have this extra dimension
                sol_compartment = sol[idx]
            dimensions_to_sum_over = tuple(range(1, sol_compartment.ndim))
            ax.plot(sol_compartment.sum(axis=dimensions_to_sum_over), label=compartment)
        fig.legend()
        if save_path:
            fig.savefig(save_path)
            with open(save_path + "_meta.txt", "x") as meta:
                meta.write(str(self.__dict__))
        return fig, ax


def build_basic_mechanistic_model(model_config):
    """
    A builder function meant to take in a model_config class and build a BasicMechanisticModel() object with defaults from the config.
    """
    return BasicMechanisticModel(
        num_age_groups=model_config.NUM_AGE_GROUPS,
        num_strains=model_config.NUM_STRAINS,
        age_limits=model_config.AGE_LIMITS,
        init_pop_size=model_config.POP_SIZE,
        birth_rate=model_config.BIRTH_RATE,
        infectious_period=model_config.INFECTIOUS_PERIOD,
        exposed_to_infectious=model_config.EXPOSED_TO_INFECTIOUS,
        vaccination_rate=model_config.VACCINATION_RATE,
        initial_infections=model_config.INITIAL_INFECTIONS,
        R0_dist=model_config.STRAIN_SPECIFIC_R0,
        num_waning_compartments=model_config.NUM_WANING_COMPARTMENTS,
        waning_protect_dist=model_config.WANING_PROTECTIONS,
        waning_time=model_config.WANING_TIME,
        num_compartments=model_config.NUM_COMPARTMENTS,
        w_idx=model_config.w_idx,
        idx=model_config.idx,
        target_population_fractions=None,
        contact_matrix=None,
        init_infection_dist=None,
    )
