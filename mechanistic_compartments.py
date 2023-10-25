import jax.config
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
from diffrax import ODETerm, SaveAt, Tsit5, diffeqsolve
from jax.random import PRNGKey
from numpyro.infer import MCMC, NUTS

import utils
from config.config_base import ConfigBase as config

numpyro.set_host_device_count(4)
jax.config.update("jax_enable_x64", True)
# plotting libraries


class BasicMechanisticModel:
    "Implementation of a basic Mechanistic model for scenario analysis"

    def __init__(self, **kwargs):
        # grab all parameters passed from config
        self.__dict__.update(kwargs)
        # if not given, generate population fractions based on observed census data
        if not self.TARGET_POPULATION_FRACTIONS:
            populations_path = (
                self.DEMOGRAPHIC_DATA
                + "population_rescaled_age_distributions/"
            )
            self.TARGET_POPULATION_FRACTIONS = utils.load_age_demographics(
                populations_path, self.REGIONS, self.AGE_LIMITS
            )["United States"]

        self.POPULATION = self.POP_SIZE * self.TARGET_POPULATION_FRACTIONS

        # if not given, load contact matrices via Dina's mixing data.
        if not self.CONTACT_MATRIX:
            self.CONTACT_MATRIX = utils.load_demographic_data(
                self.DEMOGRAPHIC_DATA,
                self.REGIONS,
                self.NUM_AGE_GROUPS,
                self.MINIMUM_AGE,
                self.AGE_LIMITS,
            )["United States"]["oth_CM"]

        # TODO does it make sense to set one and not the other if provided one ?
        if not self.INIT_WANING_DIST or not self.INIT_RECOVERED_DIST:
            sero_path = (
                self.SEROLOGICAL_DATA
                + "Nationwide_Commercial_Laboratory_Seroprevalence_Survey_20231018.csv"
            )
            pop_path = (
                self.DEMOGRAPHIC_DATA
                + "population_rescaled_age_distributions/"
            )
            (
                self.INIT_RECOVERED_DIST,
                self.INIT_WANING_DIST,
            ) = utils.load_serology_demographics(
                sero_path,
                pop_path,
                self.AGE_LIMITS,
                self.WANING_TIME,
                self.NUM_WANING_COMPARTMENTS,
                self.NUM_STRAINS,
            )

        # if not given an inital infection distribution, use max eig value vector
        if not self.INIT_INFECTION_DIST:
            eig_data = np.linalg.eig(self.CONTACT_MATRIX)
            max_index = np.argmax(eig_data[0])
            self.INIT_INFECTION_DIST = abs(eig_data[1][:, max_index])
            self.INIT_INFECTION_DIST = self.INIT_INFECTION_DIST / sum(
                self.INIT_INFECTION_DIST
            )

        # with inital infection distribution by age group, break down uniformally by number of strains.
        # TODO non-uniform strain distribution if needed.
        self.INITIAL_INFECTIONS_BY_STRAIN = (
            self.INITIAL_INFECTIONS
            * self.INIT_INFECTION_DIST[:, None]
            * np.ones(self.NUM_STRAINS)
            / self.NUM_STRAINS
        )
        init_recovered_strain_summed = np.sum(
            self.INIT_RECOVERED_DIST, axis=self.AXIS_IDX.strain
        )
        init_waning_strain_compartment_summed = np.sum(
            self.INIT_WANING_DIST,
            axis=(self.AXIS_IDX.strain, self.AXIS_IDX.wane),
        )
        inital_suseptible = (
            self.POPULATION
            - (self.INITIAL_INFECTIONS * self.INIT_INFECTION_DIST)
            - (self.POPULATION * init_recovered_strain_summed)
            - (self.POPULATION * init_waning_strain_compartment_summed)
        )
        self.INITIAL_STATE = (
            inital_suseptible,  # s
            np.zeros((self.NUM_AGE_GROUPS, self.NUM_STRAINS)),  # e
            self.INITIAL_INFECTIONS_BY_STRAIN,  # i
            (
                self.POPULATION * self.INIT_RECOVERED_DIST.transpose()
            ).transpose(),  # r
            (self.POPULATION * self.INIT_WANING_DIST.transpose()).transpose(),
        )  # w

    def get_args(self, sample=False):
        if sample:
            beta = (
                utils.sample_r0(self.STRAIN_SPECIFIC_R0)
                / self.infectious_period
            )
            waning_protections = utils.sample_waning_protections(
                self.WANING_PROTECTIONS
            )
        else:  # if we arent sampling we use values from config in self
            beta = self.STRAIN_SPECIFIC_R0 / self.INFECTIOUS_PERIOD
            waning_protections = self.WANING_PROTECTIONS

        gamma = 1 / self.INFECTIOUS_PERIOD
        sigma = 1 / self.EXPOSED_TO_INFECTIOUS
        wanning_rate = 1 / self.WANING_TIME
        # default to no cross immunity, setting diagnal to 0
        suseptibility_matrix = jnp.ones(
            (self.NUM_STRAINS, self.NUM_STRAINS)
        ) * (
            1 - jnp.diag(jnp.array([1] * self.NUM_STRAINS))
        )  # TODO use priors here
        args = [  # TODO convert to dictionary if diffeqsolve() allows for args to be a dict.
            beta,
            sigma,
            gamma,
            self.CONTACT_MATRIX,
            self.VACCINATION_RATE,
            waning_protections,
            wanning_rate,
            self.BIRTH_RATE,
            self.POPULATION,
            suseptibility_matrix,
            self.NUM_STRAINS,
            self.NUM_WANING_COMPARTMENTS,
        ]
        return args

    def incidence(self, model, incidence):
        term = ODETerm(
            lambda t, state, parameters: model(state, t, parameters)
        )
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
        numpyro.sample(
            "incidence", dist.Poisson(model_incidence), obs=incidence
        )

    def infer(self, model, incidence):
        mcmc = MCMC(
            NUTS(model, dense_mass=True),
            num_warmup=self.MCMC_NUM_WARMUP,
            num_samples=self.MCMC_NUM_SAMPLES,
            num_chains=self.MCMC_NUM_CHAINS,
            progress_bar=self.MCMC_PROGRESS_BAR,
        )
        mcmc.run(
            PRNGKey(self.MCMC_PRNGKEY),
            times=np.linspace(0.0, 100.0, 101),
            incidence=incidence,
        )
        mcmc.print_summary()

    def run(
        self,
        model,
        tf=100.0,
        save=True,
        save_path="model_run.png",
        plot_compartments=["s", "e", "i", "r", "w0", "w1", "w2", "w3"],
    ):
        """
        runs the mechanistic model using beta, gamma, sigma, and waning rate based on config file values.
        Does not sample waning protections or R0 by strain.
        """
        term = ODETerm(
            lambda t, state, parameters: model(state, t, parameters)
        )
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
            self.INITIAL_STATE,
            args=self.get_args(sample=False),
            saveat=saveat,
            max_steps=30000,
        )
        save_path = (
            save_path if save else None
        )  # dont set a save path if we dont want to save
        fig, ax = self.plot_diffrax_solution(
            solution,
            plot_compartments=plot_compartments,
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
                    self.IDX.__getitem__("W"),
                    self.W_IDX.__getitem__(compartment.strip().upper()),
                ]
                get_indexes.append(index_slice)
            else:
                get_indexes.append(
                    self.IDX.__getitem__(compartment.strip().upper())
                )

        fig, ax = plt.subplots(1)
        for compartment, idx in zip(plot_compartments, get_indexes):
            if "W" in compartment.upper():  # then idx=(idx.W, w_idx.W1/2/....)
                # if we are plotting a waning compartment, we need to parse 1 extra dimension
                sol_compartment = np.array(sol[idx[0]])[:, :, :, idx[1]]
            else:
                # non-waning compartments dont have this extra dimension
                sol_compartment = sol[idx]
            dimensions_to_sum_over = tuple(range(1, sol_compartment.ndim))
            ax.plot(
                sol_compartment.sum(axis=dimensions_to_sum_over),
                label=compartment,
            )
        fig.legend()
        if save_path:
            fig.savefig(save_path)
            with open(save_path + "_meta.txt", "x") as meta:
                meta.write(str(self.__dict__))
        return fig, ax


def build_basic_mechanistic_model(config: config):
    """
    A builder function meant to take in a Config class and build a BasicMechanisticModel() object with values from the config.
    """
    return BasicMechanisticModel(**config.__dict__)
