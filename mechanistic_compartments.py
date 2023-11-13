import json
import os
from enum import EnumMeta

import jax.config
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import pandas as pd
from diffrax import ODETerm, SaveAt, Solution, Tsit5, diffeqsolve
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

        # if not given, load population fractions based on observed census data into self
        if not self.INITIAL_POPULATION_FRACTIONS:
            self.load_inital_population_fractions()

        self.POPULATION = self.POP_SIZE * self.INITIAL_POPULATION_FRACTIONS

        # if not given, load contact matrices via mixing data into self.
        if not self.CONTACT_MATRIX:
            self.load_contact_matrix()

        # TODO does it make sense to set one and not the other if provided one ?
        # if not given, load inital waning and recovered distributions from serological data into self
        if self.INIT_WANING_DIST is None or self.INIT_RECOVERED_DIST is None:
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
        if self.INIT_INFECTED_DIST is None or self.INIT_EXPOSED_DIST is None:
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
            jnp.zeros(initial_exposed_count.shape),  # c
        )

        self.solution = None

    def get_args(
        self,
        sample: bool = False,
        sample_dist_dict: dict[str, numpyro.distributions.Distribution] = {},
    ):
        """
        A function that returns model args as a dictionary as expected by the ODETerm function f(t, y(t), args)dt
        https://docs.kidger.site/diffrax/api/terms/#diffrax.ODETerm

        for example functions f() in charge of disease dynamics see the model_odes folder.
        if sample=True, and no sample_dist_dict supplied, omicron strain beta and waning protections are automatically sampled.

        Parameters
        ----------
        `sample`: boolean
            whether or not to sample key parameters, used when model is being run in MCMC and parameters are being infered
        `sample_dist_dict`: dict(str:numpyro.distribution)
            a dictionary of parameters to sample, if empty, defaults will be sampled and rest are left at values specified in config.
            follows format "parameter_name":numpyro.Distributions.dist(). DO NOT pass numpyro.sample() objects to the dictionary.

        Returns
        ----------
        dict{str: Object}: A dictionary where key value pairs are used as parameters by an ODE model, things like R0 or contact matricies.
        """
        args = {
            "contact_matrix": self.CONTACT_MATRIX,
            "vax_rate": self.VACCINATION_RATE,
            "mu": self.BIRTH_RATE,
            "population": self.POPULATION,
            "num_strains": self.NUM_STRAINS,
            "num_waning_compartments": self.NUM_WANING_COMPARTMENTS,
            "waning_protections": self.WANING_PROTECTIONS,
        }
        if sample:
            # if user provides parameters and distributions they wish to sample, sample those
            if sample_dist_dict:
                for key, dist in sample_dist_dict.items():
                    args[key] = numpyro.sample(key, dist)
            # otherwise, by default just sample the omicron excess r0
            else:
                default_sample_dict = {}
                r0_omicron = utils.sample_r0()
                strain_specific_r0 = list(self.STRAIN_SPECIFIC_R0)
                strain_specific_r0[self.STRAIN_IDX.omicron] = r0_omicron
                default_sample_dict["r0"] = jnp.asarray(strain_specific_r0)
                args = dict(args, **default_sample_dict)

        # lets quickly update any values that depend on other parameters which may or may not be sampled.
        # set defaults if they are not in args aka not sampled.
        r0 = args.get("r0", self.STRAIN_SPECIFIC_R0)
        infectious_period = args.get(
            "infectious_period", self.INFECTIOUS_PERIOD
        )
        if "infectious_period" in args or "r0" in args:
            beta = numpyro.deterministic("beta", r0 / infectious_period)
        else:
            beta = r0 / infectious_period
        gamma = (
            1 / self.INFECTIOUS_PERIOD
            if "infectious_period" not in args
            else numpyro.deterministic("gamma", 1 / args["infectious_period"])
        )
        sigma = (
            1 / self.EXPOSED_TO_INFECTIOUS
            if "exposed_to_infectious" not in args
            else numpyro.deterministic(
                "sigma", 1 / args["exposed_to_infectious"]
            )
        )
        waning_rate = 1 / self.WANING_TIME
        # default to no cross immunity, setting diagnal to 0
        # TODO use priors informed by https://www.sciencedirect.com/science/article/pii/S2352396423002992
        suseptibility_matrix = jnp.ones(
            (self.NUM_STRAINS, self.NUM_STRAINS)
        ) * (1 - jnp.diag(jnp.array([1] * self.NUM_STRAINS)))
        # add final parameters, if your model expects added parameters, add them here
        args = dict(
            args,
            **{
                "beta": beta,
                "sigma": sigma,
                "gamma": gamma,
                "waning_rate": waning_rate,
                "susceptibility_matrix": suseptibility_matrix,
            }
        )
        return args

    def incidence(
        self,
        _,
        incidence: list[int],
        model,
        sample_dist_dict: dict[str, numpyro.distributions.Distribution] = {},
    ):
        """
        Approximate the ODE model incidence (new exposure) per time step,
        based on diffeqsolve solution obtained after self.run and sampled values of parameters.

        Parameters
        ----------
        `incidence`: list(int)
                    observed incidence of each compartment to compare against.

        `model`: function()
            an ODE style function which takes in state, time, and parameters in that order,
            and return a list of two: tuple of changes in compartment and array of incidences.

        `sample_dist_dict`: dict(str:numpyro.distribution)
            a dictionary of parameters to sample, if empty, defaults will be sampled and rest are left at values specified in config.
            follows format "parameter_name":numpyro.Distributions.dist(). DO NOT pass numpyro.sample() objects to the dictionary.

        Returns
        ----------
        List of arrays of incidence (one per time step).
        """
        solution = self.run(
            model,
            sample=True,
            sample_dist_dict=sample_dist_dict,
            tf=len(incidence),
        )
        # add 1 to strain idx because we are straified by time in the solution object
        model_incidence = jnp.sum(
            solution.ys[self.IDX.C], axis=self.AXIS_IDX.strain + 1
        )
        # axis = 0 because we take diff across time
        model_incidence = jnp.diff(model_incidence, axis=0)
        numpyro.sample(
            "incidence",
            numpyro.distributions.Poisson(model_incidence),
            obs=incidence,
        )

    def infer(
        self,
        model,
        incidence,
        sample_dist_dict: dict[str, numpyro.distributions.Distribution] = {},
        timesteps: int = 1000.0,
    ):
        """
        Runs inference given some observed incidence and a model of transmission dynamics.
        Uses MCMC and NUTS for parameter tuning of the model returns estimated parameter values given incidence.

        Parameters
        ----------
        `model`: function()
            a standard ODE style function which takes in state, time, and parameters in that order.
            for example functions see the model_odes folder.
        `incidence`: list(int)
            observed incidence of each compartment to compare against.
        `sample_dist_dict`: dict(str:numpyro.distribution)
            a dictionary of parameters to sample, if empty, defaults will be sampled and rest are left at values specified in config.
            follows format "parameter_name":numpyro.Distributions.dist(). DO NOT pass numpyro.sample() objects to the dictionary.
        `timesteps`: int
            number of timesteps over which you wish to infer over, must match len(`incidence`)
        """
        mcmc = MCMC(
            NUTS(self.incidence, dense_mass=True),
            num_warmup=self.MCMC_NUM_WARMUP,
            num_samples=self.MCMC_NUM_SAMPLES,
            num_chains=self.MCMC_NUM_CHAINS,
            progress_bar=self.MCMC_PROGRESS_BAR,
        )
        mcmc.run(
            rng_key=PRNGKey(self.MCMC_PRNGKEY),
            times=np.linspace(0.0, timesteps, int(timesteps) + 1),
            incidence=incidence,
            sample_dist_dict=sample_dist_dict,
            model=model,
        )
        mcmc.print_summary()

    def run(
        self,
        model,
        tf: int = 100.0,
        plot: bool = False,
        show: bool = False,
        save: bool = False,
        sample: bool = False,
        sample_dist_dict: dict[str, numpyro.distributions.Distribution] = {},
        save_path: str = "model_run.png",
        plot_compartments: list[str] = ["S", "E", "I", "R", "W", "C"],
    ):
        """
        Takes parameters from self and applies them to some disease dynamics modeled in `model`
        from `t0=0` to `tf`. Optionally saving compartment plots from `plot_compartments` to `save_path` if `save=True`

        Parameters
        ----------
        `model`: function()
            a standard ODE style function which takes in state, time, and parameters in that order.
            for example functions see the model_odes folder.
        `tf`: int
            stopping time point (with default configuration this is days)
        `plot`: boolean
            whether or not to plot the solution using plot_difrax_solution()
        `show`: boolean
            whether or not to show an image via matplotlib.pyplot.show()
        `save`: boolean
            whether or not to save an image and its metadata to `save_path`
        `sample`: boolean
            whether or not to sample parameters or use values from config. See `get_args()` for sampling process
        `sample_dist_dict`: dict(str:numpyro.distribution)
            a dictionary of parameters to sample, if empty, defaults will be sampled and rest are left at values specified in config.
            follows format "parameter_name":numpyro.Distributions.dist(). DO NOT pass numpyro.sample() objects to the dictionary.
        `save_path`: str
            relative or absolute path where to save an image and its metadata if `save=True`
        `plot_compartments`: list(str)
            a list of compartments to plot, strings must match those specified in self.IDX or self.W_IDX.

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
            args=self.get_args(
                sample=sample, sample_dist_dict=sample_dist_dict
            ),
            saveat=saveat,
            max_steps=30000,
        )
        self.solution = solution
        save_path = (
            save_path if save else None
        )  # dont set a save path if we dont want to save

        if plot:
            fig, ax = self.plot_diffrax_solution(
                solution,
                plot_compartments=plot_compartments,
                save_path=save_path,
            )
            if show:
                plt.show()

        return solution

    def plot_diffrax_solution(
        self,
        sol: Solution,
        plot_compartments: list[str] = ["s", "e", "i", "r", "w", "c"],
        save_path: str = None,
    ):
        """
        plots a run from diffeqsolve() with compartments `plot_compartments` returning figure and axis.
        If `save_path` is not None will save figure to that path attached with meta data in `meta_data`.

        Parameters
        ----------
        sol : difrax.Solution
            object containing ODE run as described by https://docs.kidger.site/diffrax/api/solution/
        plot_compartment : list(str), optional
            compartment titles as defined by the config file used to initialize self
        save_path : str, optional
            if `save_path = None` do not save figure to output directory. Otherwise save to relative path `save_path`
            attaching meta data of the self object.
        """
        sol = sol.ys
        get_indexes = []
        for compartment in plot_compartments:
            # if W with no index is passed, sum all W compartments together.
            if "W" == compartment.upper():
                get_indexes.append(
                    self.IDX.__getitem__(compartment.strip().upper())
                )
            # if W1/2/3/4 is supplied, we want that specific waning compartment
            elif "W" in compartment.upper():
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
            # if user selects all W compartments, we must sum across the waning axis.
            if "W" == compartment.upper():
                # the waning index + 1 because first index is for time in the solution
                sol_compartment = np.sum(
                    np.array(sol[idx]), axis=self.AXIS_IDX.wane + 1
                )
            # if W1/W2/W3... idx=(idx.W, w_idx.W1/2/....), select specific waning compartment
            elif "W" in compartment.upper():
                # if we are plotting a waning compartment, we need to parse 1 extra dimension
                sol_compartment = np.array(sol[idx[0]])[:, :, :, idx[1]]
            else:
                # non-waning compartments dont have this extra dimension
                sol_compartment = sol[idx]
            # summing over age groups + strains, 0th dim is timestep
            dimensions_to_sum_over = tuple(range(1, sol_compartment.ndim))
            ax.plot(
                sol_compartment.sum(axis=dimensions_to_sum_over),
                label=compartment,
            )
        fig.legend()
        ax.set_title(
            "Population count by compartment across all ages and strains"
        )
        ax.set_xlabel("Days since scenario start")
        ax.set_ylabel("Population Count")
        if save_path:
            fig.savefig(save_path)
            with open(save_path + "_meta.json", "w") as meta:
                self.to_json(meta)
        return fig, ax

    def plot_initial_serology(self, save_path=None, show=True):
        """
        plots a stacked bar chart representation of the initial immune compartments of the model.

        Parameters
        ----------
        save_path: {str, None}, optional
            the save path to which to save the figure, None implies figure will not be saved.
        show: {Boolean, None}, optional
            Whether or not to show the figure using plt.show() defaults to True.

        Returns
        -----------
        fig: matplotlib.figure.Figure
            Matplotlib Figure containing the generated plot
        ax: matplotlib.axes._axes.Axes
            Matplotlib axes containing data on the generated plot.
        """
        # get all recovered dists, sum across strains
        recovered_strain_sum = np.sum(
            self.INIT_RECOVERED_DIST, axis=self.AXIS_IDX.strain
        )
        # get all waned dists, sum across strains
        waned_strain_sum = np.sum(
            self.INIT_WANING_DIST, axis=self.AXIS_IDX.strain
        ).transpose()
        # combine them together into one matrix, multiply by pop counts
        immune_compartments = np.vstack(
            (recovered_strain_sum, waned_strain_sum)
        )
        immune_compartments_populations = self.POPULATION * immune_compartments
        # reverse for plot readability since we read left to right
        immune_compartments_populations = immune_compartments_populations[::-1]
        x_axis = ["R"] + ["W" + str(int(idx)) for idx in self.W_IDX]
        x_axis = x_axis[::-1]
        age_to_immunity_slice = {}
        # for each age group, plot its number of persons in each immune compartment
        # stack the bars on top of one another by summing the previous age groups underneath
        fig, ax = plt.subplots(1)
        for idx, age_group in enumerate(self.AGE_GROUP_STRS):
            age_to_immunity_slice[age_group] = immune_compartments_populations[
                :, idx
            ]
            ax.bar(
                x_axis,
                age_to_immunity_slice[age_group],
                label=age_group,
                bottom=sum(
                    [
                        age_to_immunity_slice[x]
                        for x in self.AGE_GROUP_STRS[0:idx]
                    ]
                ),
            )
        props = {"rotation": 25, "size": 7}
        plt.setp(ax.get_xticklabels(), **props)
        ax.legend()
        ax.set_title(
            "Initial Population Immunity level by %s day intervals"
            % str(self.WANING_TIME)
        )
        ax.set_xlabel("Immune Compartment")
        ax.set_ylabel("Population Count, all strains")
        if show:
            fig.show()
        if save_path:
            fig.savefig(save_path)
            with open(save_path + "_meta.json", "w") as meta:
                self.to_json(meta)
        return fig, ax

    def load_waning_and_recovered_distributions(self):
        """
        a wrapper function which loads serologically informed covid recovered and waning distributions into self, accounting for strain timing.
        Serology data initalized closely after the end of the Omicron wave on Feb 11th 2022.

        Use `self.STRAIN_IDX` to index strains in correct manner and avoid out of bounds errors

        Updates
        ----------
        self.INIT_RECOVERED_DIST : np.array
            the proportions of the total population for each age bin defined as recovered, or within 1 `self.WANING_TIME` of infection.
            has a shape of (`self.NUM_AGE_GROUPS`, `self.NUM_STRAINS`)

        self.self.INIT_WANING_DIST : np.array
            the proportions of the total population for each age bin defined as waning, or within x `self.WANING_TIME`s of infection. where x is the waning compartment
            has a shape of (`self.NUM_AGE_GROUPS`, `self.NUM_STRAINS`, `self.NUM_WANING_COMPARTMENTS`)
        """
        sero_path = (
            self.SEROLOGICAL_DATA
            + "Nationwide_Commercial_Laboratory_Seroprevalence_Survey_20231018.csv"
        )
        # pre-commit does not like pushing the sero data to the repo, so many on first run wont have it.
        if not os.path.exists(sero_path):
            # download the data from CDC website
            print(
                "seems like you are missing the serology data, lets download it from data.cdc.gov and place here here: "
                + sero_path
            )
            download_link = "https://data.cdc.gov/api/views/d2tw-32xv/rows.csv"
            sero_data = pd.read_csv(download_link)
            os.makedirs(self.SEROLOGICAL_DATA, exist_ok=True)
            sero_data.to_csv(sero_path, index=False)
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

        Updates
        ----------
        `self.INITIAL_POPULATION_FRACTIONS` : numpy.ndarray
            proportion of the total population that falls into each age group,
            length of this array is equal the number of age groups and will sum to 1.0.
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

        Updates
        ----------
        `self.CONTACT_MATRIX` : numpy.ndarray
            a matrix of shape (self.NUM_AGE_GROUPS, self.NUM_AGE_GROUPS) with each value representing TODO
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
        infected and exposed distributions, to account for people who may not be infectious yet,
        but are part of the initial infections of the model all infections assumed to be omicron.
        utilizes the ratio between gamma and sigma to determine what proportion of inital infections belong in the
        exposed (soon to be infectious), and the already infectious compartments.

        infections are stratified across age bins based on proportion of each age bin
        in individuals who have recently sero-converted before model initalization date.
        Equivalent to using proportion of each age bin in self.INIT_RECOVERED_DIST

        given that `INIT_INFECTION_DIST` = `INIT_EXPOSED_DIST` + `INIT_INFECTED_DIST`

        Updates
        ----------
        `self.INIT_INFECTION_DIST`: jnp.array(int)
            populates values using seroprevalence to produce a distribution of how new infections are
            stratified by age bin. INIT_INFECTION_DIST.shape = (self.NUM_AGE_GROUPS,) and sum(self.INIT_INFECTION_DIST) = 1.0
        `self.INIT_EXPOSED_DIST`: jnp.array(int)
            proportion of INIT_INFECTION_DIST that falls into exposed compartment, formatted into the omicron strain.
            INIT_EXPOSED_DIST.shape = (self.NUM_AGE_GROUPS, self.NUM_STRAINS)
        `self.INIT_INFECTED_DIST`: jnp.array(int)
            proportion of INIT_INFECTION_DIST that falls into infected compartment, formatted into the omicron strain.
            INIT_INFECTED_DIST.shape = (self.NUM_AGE_GROUPS, self.NUM_STRAINS)
        """
        self.INIT_INFECTION_DIST = self.INIT_RECOVERED_DIST[
            :, self.STRAIN_IDX.omicron
        ]
        # eig_data = np.linalg.eig(self.CONTACT_MATRIX)
        # max_index = np.argmax(eig_data[0])
        # self.INIT_INFECTION_DIST = abs(eig_data[1][:, max_index])
        # infections does not equal INFECTED.
        # infected is a compartment, infections means successful passing of virus
        self.INIT_INFECTION_DIST = self.INIT_INFECTION_DIST / sum(
            self.INIT_INFECTION_DIST
        )
        # old method was to use contact matrix max eigan value. produce diff values and ranking
        # [0.30490018 0.28493648 0.23049002 0.17967332] sero method 4 age bins
        # [0.27707683 0.45785665 0.1815728  0.08349373] contact matrix method, 4 bins

        # ratio of gamma / sigma defines our infected to exposed ratio at any given time
        exposed_to_infected_ratio = (
            self.EXPOSED_TO_INFECTIOUS / self.INFECTIOUS_PERIOD
        )
        self.INIT_EXPOSED_DIST = (
            exposed_to_infected_ratio * self.INIT_INFECTION_DIST
        )
        # INIT_EXPOSED_DIST is not strain stratified, put infected into the omicron strain via indicator vec
        self.INIT_EXPOSED_DIST = self.INIT_EXPOSED_DIST[:, None] * np.array(
            [0] * self.STRAIN_IDX.omicron + [1]
        )
        self.INIT_INFECTED_DIST = (
            1 - exposed_to_infected_ratio
        ) * self.INIT_INFECTION_DIST

        # INIT_INFECTED_DIST is not strain stratified, put infected into the omicron strain via indicator vec
        self.INIT_INFECTED_DIST = self.INIT_INFECTED_DIST[:, None] * np.array(
            [0] * self.STRAIN_IDX.omicron + [1]
        )

    def to_json(self, file):
        """
        a simple method which takes self.__dict__ and dumps it into `file`.
        this method effectively deals with nested numpy and jax arrays
        which are normally not JSON serializable and cause errors.

        Parameters
        ----------
        `file`: TextIOWrapper
            a file object that can be written to, usually the result of a call like open("file.txt") as f
        """

        # define a custom encoder so that things like Enums, numpy arrays,
        # and Diffrax.Solution objects can be JSON serializable
        class CustomEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray) or isinstance(obj, jnp.ndarray):
                    return obj.tolist()
                if isinstance(obj, EnumMeta):
                    return {
                        str(e): idx for e, idx in zip(obj, range(len(obj)))
                    }
                if isinstance(obj, Solution):
                    return obj.ys
                return json.JSONEncoder.default(self, obj)

        return json.dump(self.__dict__, file, indent=4, cls=CustomEncoder)


def build_basic_mechanistic_model(config: config):
    """
    A builder function meant to take in a Config class and build a BasicMechanisticModel() object with values from the config.

    Parameters
    ----------
    config : ConfigBase
        a configuration object of type `ConfigBase` or inherits from `ConfigBase`
    """
    return BasicMechanisticModel(**config.__dict__)
