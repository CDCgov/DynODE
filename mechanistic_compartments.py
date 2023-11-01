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


class BasicMechanisticModel:
    """Implementation of a basic Mechanistic model for scenario analysis,
    built by the build_basic_mechanistic_model() builder from a config file.
    for a basic runable scenario use config.config_base.py in the following way.

    from config.config_base import ConfigBase
    from mechanistic_compartments import build_basic_mechanistic_model
    `model = build_basic_mechanistic_model(ConfigBase())`
    """

    def __init__(self, **kwargs):
        """
        Initalize a basic abstract mechanistic model for covid19 case prediction.
        Should not be constructed directly, use build_basic_mechanistic_model() with a config file
        """
        # if users call __init__ instead of the builder function, kwargs will be empty, causing errors.
        assert (
            len(kwargs) > 0
        ), "Do not initalize this object without the helper function build_basic_mechanistic_model() and a config file."

        # grab all parameters passed from config
        self.__dict__.update(kwargs)

        # if not given, generate population fractions based on observed census data
        if not self.INITIAL_POPULATION_FRACTIONS:
            self.load_inital_population_fractions()

        self.POPULATION = self.POP_SIZE * self.INITIAL_POPULATION_FRACTIONS

        # if not given, load contact matrices via mixing data.
        if not self.CONTACT_MATRIX:
            self.load_contact_matrix()

        # TODO does it make sense to set one and not the other if provided one ?
        # if not given, load inital waning and recovered distributions from serological data
        if not self.INIT_WANING_DIST or not self.INIT_RECOVERED_DIST:
            self.load_waning_and_recovered_distributions()

        # because our suseptible population is not strain stratified,
        # we need to sum these inital recovered/waning distributions by their axis so shapes line up
        init_recovered_strain_summed = np.sum(
            self.INIT_RECOVERED_DIST, axis=self.AXIS_IDX.strain
        )
        init_waning_strain_compartment_summed = np.sum(
            self.INIT_WANING_DIST,
            axis=(self.AXIS_IDX.strain, self.AXIS_IDX.wane),
        )

        # if not given an inital infection distribution, use max eig value vector of contact matrix
        # disperse inital infections across infected and exposed compartments based on gamma / sigma ratio.
        if not self.INIT_INFECTED_DIST and not self.INIT_EXPOSED_DIST:
            self.load_init_infection_infected_and_exposed_dist()

        # suseptibles = Total population - infected - recovered - waning
        inital_suseptible_count = (
            self.POPULATION
            - (self.INITIAL_INFECTIONS * self.INIT_INFECTION_DIST)
            - (self.POPULATION * init_recovered_strain_summed)
            - (self.POPULATION * init_waning_strain_compartment_summed)
        )
        inital_recovered_count = (
            self.POPULATION * self.INIT_RECOVERED_DIST.transpose()
        ).transpose()
        inital_waning_count = (
            self.POPULATION * self.INIT_WANING_DIST.transpose()
        ).transpose()

        initial_infectious_count = (
            self.INITIAL_INFECTIONS * self.INIT_INFECTED_DIST
        )
        initial_exposed_count = (
            self.INITIAL_INFECTIONS * self.INIT_EXPOSED_DIST
        )
        self.INITIAL_STATE = (
            inital_suseptible_count,  # s
            initial_exposed_count,  # e
            initial_infectious_count,  # i
            inital_recovered_count,  # r
            inital_waning_count,  # w
        )

    def get_args(self, sample=False):
        """
        A function that returns model args in the correct order as expected by the ODETerm function f(t, y(t), args)dt
        https://docs.kidger.site/diffrax/api/terms/#diffrax.ODETerm

        for example functions f() in charge of disease dynamics see the model_odes folder.
        """
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
        waning_rate = 1 / self.WANING_TIME
        # default to no cross immunity, setting diagnal to 0
        # TODO use priors informed by https://www.sciencedirect.com/science/article/pii/S2352396423002992
        suseptibility_matrix = jnp.ones(
            (self.NUM_STRAINS, self.NUM_STRAINS)
        ) * (1 - jnp.diag(jnp.array([1] * self.NUM_STRAINS)))
        # if your model expects added parameters, add them here
        args = {
            "beta": beta,
            "sigma": sigma,
            "gamma": gamma,
            "contact_matrix": self.CONTACT_MATRIX,
            "vax_rate": self.VACCINATION_RATE,
            "waning_protections": waning_protections,
            "waning_rate": waning_rate,
            "mu": self.BIRTH_RATE,
            "population": self.POPULATION,
            "susceptibility_matrix": suseptibility_matrix,
            "num_strains": self.NUM_STRAINS,
            "num_waning_compartments": self.NUM_WANING_COMPARTMENTS,
        }
        return args

    def incidence(self, model, incidence):
        """
        Takes a model and some ground truth incidence,
        returning the liklihood of observing the incidence according to the model

        Parameters
        ----------
        model: function()
            a standard ODE style function which takes in state, time, and parameters in that order.
            for example functions see the model_odes folder.
        incidence: list(int)
            observed incidence of each compartment to compare against.

        Returns
        ----------
        None
        """
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
        """
        Runs inference given some observed incidence and a model of transmission dynamics.
        Uses MCMC and NUTS for parameter tuning of the model returns estimated parameter values given incidence.

        Parameters
        ----------
        model: function()
            a standard ODE style function which takes in state, time, and parameters in that order.
            for example functions see the model_odes folder.
        incidence: list(int)
            observed incidence of each compartment to compare against.
        """
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
        show=True,
        save=True,
        save_path="model_run.png",
        plot_compartments=["s", "e", "i", "r", "w0", "w1", "w2", "w3"],
    ):
        """
        Takes parameters from self and applies them to some disease dynamics modeled in `model`
        from `t0=0` to `tf`. Optionally saving compartment plots from `plot_compartments` to `save_path` if `save=True`

        Parameters
        ----------
        model: function()
            a standard ODE style function which takes in state, time, and parameters in that order.
            for example functions see the model_odes folder.
        tf: int
            stopping time point (with default configuration this is days)
        show: boolean
            whether or not to show an image via matplotlib.pyplot.show()
        save: boolean
            whether or not to save an image and its metadata to `save_path`
        save_path: str
            relative or absolute path where to save an image and its metadata if `save=True`
        plot_compartments: list(str)
            a list of compartments to plot in the image, strings may be upper or lower case and must match
            strings specified in self.IDX or self.W_IDX.

        Returns
        ----------
        Diffrax.Solution object as described by https://docs.kidger.site/diffrax/api/solution/
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
        if show:
            plt.show()
        return solution

    def plot_diffrax_solution(
        self, sol, plot_compartments=["s", "e", "i", "r"], save_path=None
    ):
        """
        plots a run from diffeqsolve() with compartments `plot_compartments` returning figure and axis.
        If `save_path` is not None will save figure to that path attached with meta data in `meta_data`.

        Parameters
        ----------
        sol : difrax.Solution
            object containing ODE run as described by https://docs.kidger.site/diffrax/api/solution/
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

    def load_waning_and_recovered_distributions(self):
        """
        a wrapper function which loads serologically informed covid recovered and waning distributions into self.
        Omicron strain will always be 3 if self.NUM_STRAINS >= 3, otherwise it will be the largest index.
        older strains like delta and alpha will be at lower indexes than omicron, 1, and 0 respectively.
        delta and alpha may be combined if self.NUM_STRAINS < 3.
        """
        sero_path = (
            self.SEROLOGICAL_DATA
            + "Nationwide_Commercial_Laboratory_Seroprevalence_Survey_20231018.csv"
        )
        pop_path = (
            self.DEMOGRAPHIC_DATA + "population_rescaled_age_distributions/"
        )
        (
            self.INIT_RECOVERED_DIST,
            self.INIT_WANING_DIST,
        ) = utils.past_infection_dist_from_serology_demographics(
            sero_path,
            pop_path,
            self.AGE_LIMITS,
            self.WANING_TIME,
            self.NUM_WANING_COMPARTMENTS,
            self.NUM_STRAINS,
        )

    def load_inital_population_fractions(self):
        """
        a wrapper function which loads age demographics for the US and sets the inital population fraction by age bin.
        """
        populations_path = (
            self.DEMOGRAPHIC_DATA + "population_rescaled_age_distributions/"
        )
        self.INITIAL_POPULATION_FRACTIONS = utils.load_age_demographics(
            populations_path, self.REGIONS, self.AGE_LIMITS
        )["United States"]

    def load_contact_matrix(self):
        """
        a wrapper function that loads a contact matrix for the USA based on mixing paterns data found here:
        https://github.com/mobs-lab/mixing-patterns
        """
        self.CONTACT_MATRIX = utils.load_demographic_data(
            self.DEMOGRAPHIC_DATA,
            self.REGIONS,
            self.NUM_AGE_GROUPS,
            self.MINIMUM_AGE,
            self.AGE_LIMITS,
        )["United States"]["avg_CM"]

    def load_init_infection_infected_and_exposed_dist(self):
        """
        loads the inital infection distribution by age, then separates infections into an
        infected and exposed distribution, all infections assumed to be omicron.
        utilizes the ratio between gamma and sigma to determine what proportion of inital infections belong in the
        exposed (soon to be infectious), and the already infectious compartments.

        infections are stratified across age bins based on proportion of each age bin
        in individuals who have recent sero-converted before model initalization date.
        Equivalent to using proportion of each age bin in self.INIT_RECOVERED_DIST

        given that `INIT_INFECTION_DIST` = `INIT_EXPOSED_DIST` + `INIT_INFECTED_DIST`
        MODIFIES
        ----------
        self.INIT_INFECTION_DIST: jnp.array(int)
            populates values using seroprevalence to produce a distribution of how infections are
            stratified by age bin. INIT_INFECTION_DIST.shape = (self.NUM_AGE_GROUPS,)

        self.INIT_EXPOSED_DIST: jnp.array(int)
            proportion of INIT_INFECTION_DIST that falls into exposed compartment, formatted into the omicron strain.
            INIT_EXPOSED_DIST.shape = (self.NUM_AGE_GROUPS, self.NUM_STRAINS)

        self.INIT_INFECTED_DIST: jnp.array(int)
            proportion of INIT_INFECTION_DIST that falls into infected compartment, formatted into the omicron strain.
            INIT_INFECTED_DIST.shape = (self.NUM_AGE_GROUPS, self.NUM_STRAINS)
        """
        # TODO initialize infections by age based on the seroprevalence by age.
        # since we are assuming similar dynamics in short time frames
        # we expect to see similar proportions of each age bin in new infections as recovered
        self.INIT_INFECTION_DIST = self.INIT_RECOVERED_DIST[
            :, self.STRAIN_IDX.omicron
        ]
        # old method was to use contact matrix max eigan value. produce diff values and ranking
        # [0.30490018 0.28493648 0.23049002 0.17967332] sero method 4 age bins
        # [0.27707683 0.45785665 0.1815728  0.08349373] contact matrix method, 4 bins
        # eig_data = np.linalg.eig(self.CONTACT_MATRIX)
        # max_index = np.argmax(eig_data[0])
        # self.INIT_INFECTION_DIST = abs(eig_data[1][:, max_index])
        # infections does not equal INFECTED.
        # infected is a compartment, infections means successful passing of virus
        self.INIT_INFECTION_DIST = self.INIT_INFECTION_DIST / sum(
            self.INIT_INFECTION_DIST
        )
        # ratio of gamma / sigma defines our infected to exposed ratio at any given time
        exposed_to_infected_ratio = (
            self.EXPOSED_TO_INFECTIOUS / self.INFECTIOUS_PERIOD
        )
        self.INIT_EXPOSED_DIST = (
            exposed_to_infected_ratio * self.INIT_INFECTION_DIST
        )
        self.INIT_EXPOSED_DIST = self.INIT_EXPOSED_DIST[:, None] * np.array(
            [0] * self.STRAIN_IDX.omicron + [1]
        )
        self.INIT_INFECTED_DIST = (
            1 - exposed_to_infected_ratio
        ) * self.INIT_INFECTION_DIST

        self.INIT_INFECTED_DIST = self.INIT_INFECTED_DIST[:, None] * np.array(
            [0] * self.STRAIN_IDX.omicron + [1]
        )


def build_basic_mechanistic_model(config: config):
    """
    A builder function meant to take in a Config class and build a BasicMechanisticModel() object with values from the config.
    """
    return BasicMechanisticModel(**config.__dict__)
