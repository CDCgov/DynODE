import datetime
import json
import os
import subprocess
import warnings
from enum import EnumMeta
from functools import partial

import jax.config
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import pandas as pd
from diffrax import (
    ODETerm,
    PIDController,
    SaveAt,
    Solution,
    Tsit5,
    diffeqsolve,
)
from jax import jit
from jax.random import PRNGKey
from jax.scipy.stats.norm import pdf
from numpyro import distributions as Dist
from numpyro.infer import MCMC, NUTS
from PIL import Image
from PIL.PngImagePlugin import PngInfo

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
        initialize a basic abstract mechanistic model for covid19 case prediction.
        Should not be constructed directly, use build_basic_mechanistic_model() with a config file
        """
        # if users call __init__ instead of the builder function, kwargs will be empty, causing errors.
        assert (
            len(kwargs) > 0
        ), "Do not initialize this object without the helper function build_basic_mechanistic_model() and a config file."

        # grab all parameters passed from config
        self.__dict__.update(kwargs)
        self.config_file = kwargs

        # GENERATE CROSS IMMUNITY MATRIX with protection from STRAIN_INTERACTIONS most recent infected strain.
        if self.CROSSIMMUNITY_MATRIX is None:
            self.build_cross_immunity_matrix()
        # if not given, load population fractions based on observed census data into self
        if not self.INITIAL_POPULATION_FRACTIONS:
            self.load_initial_population_fractions()

        self.POPULATION = self.POP_SIZE * self.INITIAL_POPULATION_FRACTIONS
        # self.POPULATION.shape = (NUM_AGE_GROUPS,)

        # if not given, load contact matrices via mixing data into self.
        if not self.CONTACT_MATRIX:
            self.load_contact_matrix()
        # self.CONTACT_MATRIX.shape = (NUM_AGE_GROUPS, NUM_AGE_GROUPS)

        if self.INIT_IMMUNE_HISTORY is None:
            self.load_immune_history_via_abm()
        # self.INIT_IMMUNE_HISTORY.shape = (age, hist, num_vax, waning)

        # disperse inital infections across infected and exposed compartments based on gamma / sigma ratio.
        # stratify initial infections appropriately across age, hist, vax counts
        if self.INIT_INFECTED_DIST is None or self.INIT_EXPOSED_DIST is None:
            self.load_init_infection_infected_and_exposed_dist_via_abm()
        # self.INIT_INFECTION_DIST.shape = (age, hist, num_vax, strain)

        if self.VAX_MODEL_KNOTS is None:
            self.load_vaccination_model()
        # loads params used in self.vaccination_rate()

        if self.EXTERNAL_I_DISTRIBUTIONS is None:
            self.load_external_i_distributions()
        # loads params used in self.external_i()

        initial_infectious_count = (
            self.INITIAL_INFECTIONS * self.INIT_INFECTED_DIST
        )
        initial_infectious_count_ages = jnp.sum(
            initial_infectious_count,
            axis=(
                self.I_AXIS_IDX.hist,
                self.I_AXIS_IDX.vax,
                self.I_AXIS_IDX.strain,
            ),
        )
        initial_exposed_count = (
            self.INITIAL_INFECTIONS * self.INIT_EXPOSED_DIST
        )
        initial_exposed_count_ages = jnp.sum(
            initial_exposed_count,
            axis=(
                self.I_AXIS_IDX.hist,
                self.I_AXIS_IDX.vax,
                self.I_AXIS_IDX.strain,
            ),
        )
        # suseptible / partial susceptible = Total population - infected - exposed
        initial_suseptible_count = (
            self.POPULATION
            - initial_infectious_count_ages
            - initial_exposed_count_ages
        )[:, np.newaxis, np.newaxis, np.newaxis] * self.INIT_IMMUNE_HISTORY

        self.INITIAL_STATE = (
            initial_suseptible_count,  # s
            initial_exposed_count,  # e
            initial_infectious_count,  # i
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
        if sample=True, and no sample_dist_dict supplied, infectious period and BA1.1 introduction time are automatically sampled.

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
            "CONTACT_MATRIX": self.CONTACT_MATRIX,
            "POPULATION": self.POPULATION,
            "NUM_STRAINS": self.NUM_STRAINS,
            "NUM_AGE_GROUPS": self.NUM_AGE_GROUPS,
            "NUM_WANING_COMPARTMENTS": self.NUM_WANING_COMPARTMENTS,
            "WANING_PROTECTIONS": self.WANING_PROTECTIONS,
            "MAX_VAX_COUNT": self.MAX_VAX_COUNT,
            "CROSSIMMUNITY_MATRIX": self.CROSSIMMUNITY_MATRIX,
            "VAX_EFF_MATRIX": self.VAX_EFF_MATRIX,
        }
        if sample:
            # if user provides parameters and distributions they wish to sample, sample those
            # otherwise, we simply sample infectious period and introduction times by default
            if not sample_dist_dict:
                sample_dist_dict = {
                    # warning: decreasing floor of INFECTIOUS_PERIOD causes more steps to be taken by solvers
                    "INFECTIOUS_PERIOD": Dist.TruncatedNormal(
                        loc=10, scale=2, low=1.0
                    ),
                }
                # introduction times are used by a different function, self.external_i, which is just in time compiled
                # thus we set it as a part of self, rather than in args dict.
                # if sampling is needed then
                self.INTRODUCTION_TIMES_SAMPLE = [
                    numpyro.sample(
                        "INTRODUCTION_TIME_{}".format(i),
                        Dist.TruncatedNormal(loc=intro_time, scale=20, low=10),
                    )
                    for i, intro_time in enumerate(self.INTRODUCTION_TIMES)
                ]
            # either using the default sample_dist_dict, or the one provided by the user
            # transform these distributions into numpyro samples.
            for key, item in sample_dist_dict.items():
                # sometimes you may want to sample the elements of a list, like R0 for strains
                # check for that here:
                if isinstance(item, list):
                    # build up a list of samples
                    sample_list = jnp.zeros(shape=(len(item),))
                    for i, dist in enumerate(item):
                        # sometimes people pass a mixture of static and sampled values. check for numbers
                        if isinstance(dist, (int, float)) and not isinstance(
                            dist, bool
                        ):
                            sample = numpyro.deterministic(
                                key + "_" + str(i), dist
                            )
                        else:
                            sample = numpyro.sample(key + "_" + str(i), dist)
                        sample_list = sample_list.at[i].set(sample)
                    args[key] = sample_list
                else:
                    args[key] = numpyro.sample(key, item)

        # lets quickly update any values that depend on other parameters which may or may not be sampled.
        # set defaults if they are not in args aka not sampled.
        r0 = args.get("R0", self.STRAIN_SPECIFIC_R0)
        infectious_period = args.get(
            "INFECTIOUS_PERIOD", self.INFECTIOUS_PERIOD
        )
        if "INFECTIOUS_PERIOD" in args or "R0" in args:
            beta = numpyro.deterministic("BETA", r0 / infectious_period)
        else:
            beta = r0 / infectious_period
        gamma = (
            1 / self.INFECTIOUS_PERIOD
            if "INFECTIOUS_PERIOD" not in args
            else numpyro.deterministic("gamma", 1 / args["INFECTIOUS_PERIOD"])
        )
        sigma = (
            1 / self.EXPOSED_TO_INFECTIOUS
            if "EXPOSED_TO_INFECTIOUS" not in args
            else numpyro.deterministic(
                "SIGMA", 1 / args["EXPOSED_TO_INFECTIOUS"]
            )
        )
        # since our last waning time is zero to account for last compartment never waning
        # we include an if else statement to catch a division by zero error here.
        waning_rates = [
            1 / waning_time if waning_time > 0 else 0
            for waning_time in self.WANING_TIMES
        ]
        # add final parameters, if your model expects added parameters, add them here
        args = dict(
            args,
            **{
                "BETA": beta,
                "SIGMA": sigma,
                "GAMMA": gamma,
                "WANING_RATES": waning_rates,
                "EXTERNAL_I": partial(self.external_i),
                "VACCINATION_RATES": self.vaccination_rate,
                "BETA_COEF": self.beta_coef,
            }
        )
        return args

    @partial(jit, static_argnums=(0))
    def external_i(self, t):
        """
        Given some time t, returns jnp.array of shape self.INITIAL_STATE[self.IDX.I] representing external infected persons
        interacting with the population. it does so by calling some function f_s(t) for each strain s.

        MUST BE CONTINUOUS AND DIFFERENTIABLE FOR ALL TIMES t.

        Parameters
        ----------
        `t`: float as Traced<ShapedArray(float32[])>
            current time in the model, due to the just-in-time nature of Jax this float value may be contained within a
            traced array of shape () and size 1. Thus no explicit comparison should be done on "t".

        Returns
        -----------
        external_i_compartment: jnp.array()
            jnp.array(shape=(self.INITIAL_STATE[self.IDX.I].shape)) of external individuals to the system
            interacting with susceptibles within the system, used to impact force of infection.
        """
        # set up our return value
        external_i_compartment = jnp.zeros(
            self.INITIAL_STATE[self.IDX.I].shape
        )
        # default from the config
        external_i_distributions = self.EXTERNAL_I_DISTRIBUTIONS
        # pick sampled versions or defaults from config
        if hasattr(self, "INTRODUCTION_TIMES_SAMPLE"):
            # if we are sampling, sample the introduction times and use it to inform our
            # external_i_distribution as the mean distribution day.
            for introduced_strain_idx, introduced_time_sampler in enumerate(
                self.INTRODUCTION_TIMES_SAMPLE
            ):
                dist_idx = self.NUM_STRAINS - introduced_strain_idx - 1
                # use a normal PDF with std dv
                external_i_distributions[dist_idx] = partial(
                    pdf, loc=introduced_time_sampler, scale=7
                )
        introduction_age_mask = jnp.where(
            jnp.array(self.INTRODUCTION_AGE_MASK),
            1,
            0,
        )
        for strain in self.STRAIN_IDX:
            external_i_distribution = external_i_distributions[strain]
            external_i_compartment = external_i_compartment.at[
                introduction_age_mask, 0, 0, strain
            ].set(
                external_i_distribution(t)
                * self.INTRODUCTION_PERCENTAGE
                * self.POPULATION[self.INTRODUCTION_AGE_MASK]
            )
        return external_i_compartment

    @partial(jit, static_argnums=(0))
    def vaccination_rate(self, t):
        """
        Given some time t, returns a jnp.array of shape (self.NUM_AGE_GROUPS, self.MAX_VAX_COUNT + 1)
        representing the age / vax history stratified vaccination rates for an additional vaccine. Used by transmission models
        to determine vaccination rates at a particular time step.

        MUST BE CONTINUOUS AND DIFFERENTIABLE FOR ALL TIMES t.

        Parameters
        ----------
        t: float as Traced<ShapedArray(float32[])>
            current time in the model, due to the just-in-time nature of Jax this float value may be contained within a
            traced array of shape () and size 1. Thus no explicit comparison should be done on "t".

        Returns
        -----------
        vaccination_rates: jnp.array()
            jnp.array(shape=(self.NUM_AGE_GROUPS, self.MAX_VAX_COUNT + 1)) of vaccination rates for each age bin and vax history strata.
        """
        return jnp.exp(
            utils.VAX_FUNCTION(
                t,
                self.VAX_MODEL_KNOT_LOCATIONS,
                self.VAX_MODEL_BASE_EQUATIONS,
                self.VAX_MODEL_KNOTS,
            )
        )

    def beta_coef(self, t):
        """Returns a coefficient for the beta value for cases of seasonal forcing or external impacts
        onto beta not direclty measured in the model. EG: masking mandates or holidays.
        Currently implemented via an array search with timings BETA_TIMES and coefficients BETA_COEFICIENTS

        Parameters
        ----------
        t: float as Traced<ShapedArray(float32[])>
            current time in the model, due to the just-in-time nature of Jax this float value may be contained within a
            traced array of shape () and size 1. Thus no explicit comparison should be done on "t".

        Returns:
        coefficient with which BETA can be multiplied with to externally increase or decrease the value to account for measures or seasonal forcing.
        """
        # this is basically a smart lookup function that works with JAX just in time compilation
        return self.BETA_COEFICIENTS[
            jnp.maximum(0, jnp.searchsorted(self.BETA_TIMES, t) - 1)
        ]

    def incidence(
        self,
        incidence: list[int],
        model,
        negbin=True,
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
        # add 1 to idxs because we are straified by time in the solution object
        # sum down to just time x age bins
        model_incidence = jnp.sum(
            solution.ys[self.IDX.C],
            axis=(
                self.I_AXIS_IDX.hist + 1,
                self.I_AXIS_IDX.vax + 1,
                self.I_AXIS_IDX.strain + 1,
            ),
        )
        # axis = 0 because we take diff across time
        model_incidence = jnp.diff(model_incidence, axis=0)

        # sample infection hospitalization rate here
        with numpyro.plate("num_age", self.NUM_AGE_GROUPS):
            ihr = numpyro.sample("ihr", Dist.Beta(0.5, 10))

        # scale model_incidence w ihr and apply Poisson or NB observation model
        if negbin:
            k = numpyro.sample("k", Dist.HalfCauchy(1.0))
            numpyro.sample(
                "incidence",
                Dist.NegativeBinomial2(
                    mean=model_incidence * ihr, concentration=k
                ),
                obs=incidence,
            )
        else:
            numpyro.sample(
                "incidence",
                Dist.Poisson(model_incidence * ihr),
                obs=incidence,
            )

    def infer(
        self,
        model,
        incidence: list,
        sample_dist_dict: dict[str, numpyro.distributions.Distribution] = {},
        negbin: bool = True,
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
            NUTS(
                self.incidence,
                dense_mass=True,
                max_tree_depth=5,
                init_strategy=numpyro.infer.init_to_median,
            ),
            num_warmup=self.MCMC_NUM_WARMUP,
            num_samples=self.MCMC_NUM_SAMPLES,
            num_chains=self.MCMC_NUM_CHAINS,
            progress_bar=self.MCMC_PROGRESS_BAR,
        )
        mcmc.run(
            rng_key=PRNGKey(self.MCMC_PRNGKEY),
            incidence=incidence,
            model=model,
            negbin=negbin,
            sample_dist_dict=sample_dist_dict,
        )
        mcmc.print_summary()

    def run(
        self,
        model,
        tf: int = 100.0,
        show: bool = False,
        save: bool = False,
        sample: bool = False,
        sample_dist_dict: dict[str, numpyro.distributions.Distribution] = {},
        save_path: str = "model_run.png",
        plot_commands: list[str] = ["S", "E", "I", "C"],
        log_scale: bool = False,
    ):
        """
        Takes parameters from self and applies them to some disease dynamics modeled in `model`
        from `t0=0` to `tf`. Optionally saving compartment plots from `plot_commands` to `save_path` if `save=True`

        Parameters
        ----------
        `model`: function()
            a standard ODE style function which takes in state, time, and parameters in that order.
            for example functions see the model_odes folder.
        `tf`: int
            stopping time point (with default configuration this is days)
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
        `plot_commands`: list(str)
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
        dt0 = 1.0
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
            # discontinuities due to beta manipulation specified as jump_ts
            stepsize_controller=PIDController(
                rtol=1e-3,
                atol=1e-6,
                jump_ts=list(self.BETA_TIMES),
            ),
            saveat=saveat,
            # higher for large time scales / rapid changes
            max_steps=int(1e6),
        )
        self.solution = solution
        save_path = (
            save_path if save else None
        )  # dont set a save path if we dont want to save

        if show or save:
            fig, ax = self.plot_diffrax_solution(
                solution,
                plot_commands=plot_commands,
                save_path=save_path,
                log_scale=log_scale,
            )
            if show:
                plt.show()

        return solution

    def plot_diffrax_solution(
        self,
        sol: Solution,
        plot_commands: list[str] = ["S", "E", "I", "C"],
        save_path: str = None,
        log_scale: bool = False,
        fig: plt.figure = None,
        ax: plt.axis = None,
    ):
        """
        plots a run from diffeqsolve() with `plot_commands` returning figure and axis.
        If `save_path` is not None will save figure to that path attached with meta data in `meta_data`.

        Parameters
        ----------
        sol : difrax.Solution
            object containing ODE run as described by https://docs.kidger.site/diffrax/api/solution/
        plot_commands : list(str), optional
            commands to the plotter on which populations to show, may be compartment titles, strain names, waning compartments, or explicit numpy slices!
            see utils/get_timeline_from_solution_with_command() for more in depth explanation of commands.
        save_path : str, optional
            if `save_path = None` do not save figure to output directory. Otherwise save to relative path `save_path`
            attaching meta data of the self object.
        fig: matplotlib.pyplot.figure
            if this plot is part of a larger subplots, pass the figure object here, otherwise one is created
        ax: matplotlib.pyplot.axis
            if this plot is part of a larger subplots, pass the specific axis object here, otherwise one is created

        Returns
        ----------
        fig, ax : matplotlib.Figure/axis object
            objects containing the matplotlib figure and axis for further modifications if needed.
        """
        plot_commands = [x.strip() for x in plot_commands]
        sol = sol.ys
        if fig is None or ax is None:
            fig, ax = plt.subplots(1)
            ax.set_title(
                "Population count by compartment across all ages and strains"
            )
        for command in plot_commands:
            timeline, label = utils.get_timeline_from_solution_with_command(
                sol,
                self.IDX,
                self.W_IDX,
                self.STRAIN_IDX,
                command,
            )
            days = list(range(len(timeline)))
            # incidence is aggregated weekly, so our array increases 7 days at a time
            if command == "incidence":
                days = [day * 7 for day in days]
            x_axis = [
                self.INIT_DATE + datetime.timedelta(days=day) for day in days
            ]
            if command == "incidence":
                ax.scatter(x_axis, timeline, label=label, s=1)
            else:
                ax.plot(
                    x_axis,
                    timeline,
                    label=label,
                )
        ax.tick_params(axis="x", labelrotation=45)
        ax.legend()
        ax.set_ylabel("Population Count")
        if log_scale:
            ax.set_yscale("log")
        if save_path:
            metadata = PngInfo()
            if self.GIT_REPO.is_dirty():
                warnings.warn(
                    """\n Uncommitted Changes Warning: In order to ensure replicability of your image,
                    please commit/push your changes so that the commit
                    hash may be saved in the meta data of this image, along with config parameters. \n
                    Reproducibility is pivotal to science!"""
                )
            metadata.add_text("model", self.to_json())
            fig.savefig(save_path, pil_kwargs={"pnginfo": metadata})
        return fig, ax

    def plot_initial_serology(self, save_path: str = None, show: bool = True):
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
        # combine them together into one matrix, multiply by pop counts
        immune_compartments = [
            self.INIT_IMMUNE_HISTORY[:, :, :, w_idx] for w_idx in self.W_IDX
        ]
        immune_compartments_populations = self.POPULATION * immune_compartments
        # reverse for plot readability since we read left to right
        immune_compartments_populations = immune_compartments_populations[::-1]
        x_axis = ["W" + str(int(idx)) for idx in self.W_IDX][::-1]
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
        ax.set_title("Initial Population Immunity level by waning compartment")
        ax.set_xlabel("Immune Compartment")
        ax.set_ylabel("Population Count, all strains")
        if show:
            fig.show()
        if save_path:
            metadata = PngInfo()
            for key, val in self.__dict__.items():
                metadata.add_text(key, str(val))
            fig.savefig(save_path, pil_kwargs={"pnginfo": metadata})
        return fig, ax

    def load_immune_history_via_serology(self):
        """
        a wrapper function which loads serologically informed covid immune history distributions into self, accounting for strain timing.
        Serology data initalized closely after the end of the Omicron wave on Feb 11th 2022. Individuals are marked with
        previous omicron exposure, or previous non-omicron exposure, as well number of vaccinations. Placed into waning compartments
        according to the more recent of exposure or vaccination.

        Use `self.STRAIN_IDX` to index strains in correct manner and avoid out of bounds errors

        Updates
        ----------
        self.self.INIT_IMMUNE_HISTORY : np.array
            the proportions of the total population for each age bin defined as waning, or within x `self.WANING_TIME`s of infection. where x is the waning compartment
            has a shape of (`self.NUM_AGE_GROUPS`, `self.NUM_PREV_INF_HIST`, `self.MAX_VAX_COUNT + 1`, `self.NUM_WANING_COMPARTMENTS`)
        """
        warnings.warn("DEPRECATED WARNING: function is deprecated")
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
        # return the proportions of each age group with each immune history
        # immune history = natural infection + vaccination tracking whats more recent.
        self.INIT_IMMUNE_HISTORY = (
            utils.past_immune_dist_from_serology_demographics(
                sero_path,
                pop_path,
                self.AGE_LIMITS,
                self.WANING_TIMES,
                self.NUM_WANING_COMPARTMENTS,
                self.MAX_VAX_COUNT,
                self.NUM_STRAINS,
            )
        )

    def load_immune_history_via_abm(self):
        self.INIT_IMMUNE_HISTORY = utils.past_immune_dist_from_abm(
            self.SIM_DATA,
            self.NUM_AGE_GROUPS,
            self.AGE_LIMITS,
            self.MAX_VAX_COUNT,
            self.WANING_TIMES,
            self.NUM_WANING_COMPARTMENTS,
            self.NUM_STRAINS,
            self.STRAIN_IDX,
        )

    def load_initial_population_fractions(self):
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

    def load_init_infection_infected_and_exposed_dist_via_serology(self):
        """
        loads the inital infection distribution by age, then separates infections into an
        infected and exposed distributions, to account for people who may not be infectious yet,
        but are part of the initial infections of the model all infections assumed to be omicron.
        utilizes the ratio between gamma and sigma to determine what proportion of inital infections belong in the
        exposed (soon to be infectious), and the already infectious compartments.

        infections are stratified across age bins based on proportion of each age bin
        in individuals who have recently sero-converted before model initalization date.
        Equivalent to using proportion of each age bin in top waning compartment

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
        warnings.warn("DEPRECATED WARNING: function is deprecated")
        # base infection distribution on the currently freshly recovered individuals.
        # TODO this is a problem because we dont know if those in 0 waning are there due to nat
        # infection or there due to vaccination. So what we will do is pick those who are infected with omicron
        # since we know that this model is initalized just after omicron wave, so its likely that they have
        # just been infected naturally and are in wane=0 because of that rather than vaccination.
        infection_shape = tuple(
            list(self.INIT_IMMUNE_HISTORY.shape)[:-1] + [self.NUM_STRAINS]
        )  # age, state, num_vax, strain
        self.INIT_INFECTION_DIST = np.zeros(infection_shape)
        # TODO as of right now we only infect those in states with omicron, what about
        # fully susceptible / non-omicron exposure people????
        # we need some distribution for infections across state within an age group.
        states_with_omicron = utils.all_immune_states_with(
            self.STRAIN_IDX.omicron, self.NUM_STRAINS
        )
        self.INIT_INFECTION_DIST[
            :, states_with_omicron, :, self.STRAIN_IDX.omicron
        ] = self.INIT_IMMUNE_HISTORY[:, states_with_omicron, :, 0]
        # infections does not equal INFECTED.
        # infected is a compartment, infections means successful passing of virus
        self.INIT_INFECTION_DIST = self.INIT_INFECTION_DIST / np.sum(
            self.INIT_INFECTION_DIST, axis=(0, 1, 2, 3)
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
        # an array used to add the 'strain' dimension into exposed and infected arrays.
        # strain_filler_array = np.array(  # build strain array
        #     [0] * self.STRAIN_IDX.omicron
        #     + [1]
        #     + [0] * (self.NUM_STRAINS - 1 - self.STRAIN_IDX.omicron)
        # )
        # INIT_EXPOSED_DIST is not strain stratified, put infected into the omicron strain via indicator vec
        # self.INIT_EXPOSED_DIST = (
        #     self.INIT_EXPOSED_DIST[:, :, :, None] * strain_filler_array
        # )
        # next we correct for the states we cut out.
        # self.INIT_EXPOSED_DIST = np.concatenate(
        #     [self.INIT_EXPOSED_DIST, np.zeros((self.INIT_EXPOSED_DIST.shape))],
        #     axis=1,
        # )
        self.INIT_INFECTED_DIST = (
            1 - exposed_to_infected_ratio
        ) * self.INIT_INFECTION_DIST

        # INIT_INFECTED_DIST is not strain stratified, put infected into the omicron strain via indicator vec
        # self.INIT_INFECTED_DIST = (
        #     self.INIT_INFECTED_DIST[:, :, :, None] * strain_filler_array
        # )

    def load_init_infection_infected_and_exposed_dist_via_abm(self):
        """
        loads the inital infection distribution by age, then separates infections into an
        infected and exposed distributions, to account for people who may not be infectious yet,
        but are part of the initial infections of the model all infections assumed to be omicron.
        utilizes the ratio between gamma and sigma to determine what proportion of inital infections belong in the
        exposed (soon to be infectious), and the already infectious compartments.

        given that `INIT_INFECTION_DIST` = `INIT_EXPOSED_DIST` + `INIT_INFECTED_DIST`

        Updates
        ----------
        `self.INIT_INFECTION_DIST`: jnp.array(int)
            populates values using abm to produce a distribution of how new infections are
            stratified by age bin, vax, immune_history, and strain strata. All new infections are classified in STRAIN_IDX.omicron
        `self.INIT_EXPOSED_DIST`: jnp.array(int)
            proportion of INIT_INFECTION_DIST that falls into exposed compartment,
            stratified by age bin, vax, immune_history, and strain strata. All new exposures are classified in STRAIN_IDX.omicron
        `self.INIT_INFECTED_DIST`: jnp.array(int)
            proportion of INIT_INFECTION_DIST that falls into infected compartment,
            stratified by age bin, vax, immune_history, and strain strata. All new infected are classified in STRAIN_IDX.omicron
        `self.INITIAL_INFECTIONS`: float
            if `INITIAL_INFECTIONS` is not specified in the config, will use the proportion of the total population
            that is exposed or infected in the abm, multiplied by the population size, for the models number of infections.
        """
        (
            self.INIT_INFECTION_DIST,
            self.INIT_EXPOSED_DIST,
            self.INIT_INFECTED_DIST,
            proportion_infected,
        ) = utils.init_infections_from_abm(
            self.SIM_DATA,
            self.NUM_AGE_GROUPS,
            self.AGE_LIMITS,
            self.MAX_VAX_COUNT,
            self.WANING_TIMES,
            self.NUM_STRAINS,
            self.STRAIN_IDX,
        )
        if self.INITIAL_INFECTIONS is None:
            self.INITIAL_INFECTIONS = self.POP_SIZE * proportion_infected

    def build_cross_immunity_matrix(self):
        """
        Loads the Crossimmunity matrix given the strain interactions matrix.
        Strain interactions matrix is a matrix of shape (num_strains, num_strains) representing the relative immune escape risk
        of those who are being challenged by a strain in dim 0 but have recovered from a strain in dim 1.
        Neither the strain interactions matrix nor the crossimmunity matrix take into account waning.
        Updates
        ----------
        self.CROSSIMMUNITY_MATRIX:
            updates this matrix to shape (self.NUM_STRAINS, self.NUM_PREV_INF_HIST) containing the relative immune escape
            values for each challenging strain compared to each prior immune history in the model.
        """
        self.CROSSIMMUNITY_MATRIX = utils.strain_interaction_to_cross_immunity(
            self.NUM_STRAINS, self.STRAIN_INTERACTIONS
        )

    def load_vaccination_model(self):
        """
        loads parameters of a polynomial spline vaccination model stratified on age bin and current vaccination status.
        also loads in the spline knot locations.
        """
        parameters = pd.read_csv(self.VAX_MODEL_DATA)
        age_bins = len(parameters["age_group"].unique())
        vax_bins = len(parameters["dose"].unique())
        # change this if you start using higher degree polynomials to fit vax model
        assert age_bins == self.NUM_AGE_GROUPS, (
            "the number of age bins in your model does not match the input vaccination parameters, "
            + "please provide your own vaccination parameters that match, or adjust your age bins"
        )

        assert vax_bins == self.MAX_VAX_COUNT + 1, (
            "the number of vaccination counts in your model does not match the input vaccination parameters, "
            + "please provide your own vaccination parameters that match, or adjust your age bins"
        )
        vax_knots = np.zeros((age_bins, vax_bins, self.VAX_MODEL_NUM_KNOTS))
        vax_knot_locations = np.zeros(
            (age_bins, vax_bins, self.VAX_MODEL_NUM_KNOTS)
        )
        vax_base_equations = np.zeros((age_bins, vax_bins, 4))  # always 4
        for row in parameters.itertuples():
            _, age_group, vaccination = row[0:3]
            intersect_and_ts = row[3:7]
            knot_coefficients = row[7 : 7 + self.VAX_MODEL_NUM_KNOTS]
            knot_locations = row[7 + self.VAX_MODEL_NUM_KNOTS :]
            age_group_idx = self.AGE_GROUP_IDX[age_group]
            vax_idx = vaccination - 1
            vax_base_equations[age_group_idx, vax_idx, :] = np.array(
                intersect_and_ts
            )
            vax_knots[age_group_idx, vax_idx, :] = np.array(knot_coefficients)
            vax_knot_locations[age_group_idx, vax_idx, :] = np.array(
                knot_locations
            )
        self.VAX_MODEL_KNOTS = jnp.array(vax_knots)
        self.VAX_MODEL_KNOT_LOCATIONS = jnp.array(vax_knot_locations)
        self.VAX_MODEL_BASE_EQUATIONS = jnp.array(vax_base_equations)

    def load_external_i_distributions(self):
        """
        a function that loads external_i_distributions array into the model.
        this list of functions dictate the number of infected individuals EXTERNAL TO THE POPULATION are introduced at a particular timestep.

        each function within this list must be differentiable at all input values `t`>=0 and return some value such that
        sum(f(t)) forall t>=0 = 1.0. By default we use a normal PDF to approximate this value.

        Updates
        ----------
        EXTERNAL_I_DISTRIBUTIONS: list[func(jac_tracer(float))->float]
        updates each strain to have its own introduction function, centered around the corresponding introduction time in self.INTRODUCTION_TIMES
        historical strains, which are introduced before model initialization are given the zero function f(_) -> 0.
        """

        def zero_function(_):
            return 0

        self.EXTERNAL_I_DISTRIBUTIONS = [
            zero_function for _ in range(self.NUM_STRAINS)
        ]
        for introduced_strain_idx, introduced_time in enumerate(
            self.INTRODUCTION_TIMES
        ):
            # earlier introduced strains earlier will be placed closer to historical strains (0 and 1)
            dist_idx = (
                self.NUM_STRAINS
                - self.NUM_INTRODUCED_STRAINS
                + introduced_strain_idx
            )
            # use a normal PDF with std dv
            self.EXTERNAL_I_DISTRIBUTIONS[dist_idx] = partial(
                pdf, loc=introduced_time, scale=7
            )

    def to_json(self, file=None):
        """
        a simple method which takes self.config_file and dumps it into `file`.
        this method effectively deals with nested numpy and jax arrays
        which are normally not JSON serializable and cause errors.
        Also able to return a string representation of the model JSON if `file` is None

        Parameters
        ----------
        `file`: TextIOWrapper | None
            a file object that can be written to, usually the result of a call like open("file.txt")

        Returns
        ----------
        None if file object is passed as parameter, str of JSON otherwise.
        """

        # define a custom encoder so that things like Enums, numpy arrays,
        # and Diffrax.Solution objects can be JSON serializable.
        # Wrap everything in a dict with the object type inside.
        class CustomEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray) or isinstance(obj, jnp.ndarray):
                    return {"type": "jax", "val": obj.tolist()}
                if isinstance(obj, EnumMeta):
                    return {
                        "type": "enum",
                        "val": {
                            str(e): idx for e, idx in zip(obj, range(len(obj)))
                        },
                    }
                if isinstance(obj, Solution):
                    return {"type": "sol", "val": obj.ys}
                if isinstance(obj, datetime.date):
                    return {"type": "date", "val": obj.strftime("%d-%m-%y")}
                try:
                    res = {
                        "type": "default",
                        "val": json.JSONEncoder.default(self, obj),
                    }
                except TypeError:
                    res = "error not serializable"
                return res

        if file:
            return json.dump(
                self.config_file, file, indent=4, cls=CustomEncoder
            )
        else:  # if given empty file, just return JSON string
            return json.dumps(self.config_file, indent=4, cls=CustomEncoder)


def build_basic_mechanistic_model(config: config):
    """
    A builder function meant to take in a Config class and build a BasicMechanisticModel() object with values from the config.

    Parameters
    ----------
    config : ConfigBase
        a configuration object of type `ConfigBase` or inherits from `ConfigBase`

    Returns
    ----------
    BasicMechanisticModel with parameters from the config file.
    """
    return BasicMechanisticModel(**config.__dict__)


def build_model_from_figure(im_path: str):
    """
    A builder function meant to take in an image and associated meta data and reconstruct the model that generated the image.

    Note: if you wish to ensure full reproducibility you must also revert your repository to the commit hash printed by this function.

    Parameters
    ----------
    im_path: str
        path to image you wish to copy model parameters from

    Returns
    ----------
    BasicMechanisticModel with parameters from the meta data of the image. Returns None if meta data does not exist.
    """
    im = Image.open(im_path)
    metadata = im.text
    if not metadata or not metadata["model"]:
        print(
            "image passed does not contain metadata or is incorrect format, unable to recreate associated model"
        )
        return None
    metadata = metadata["model"]
    image_config = config(**utils.from_json(metadata))
    if not hasattr(image_config, "GIT_HASH"):
        print(
            "metadata does not contain git hash, code associated with this image irretrievable"
        )
    else:
        cur_git_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("ascii")
            .strip()
        )
        print("current git hash of codebase: " + str(cur_git_hash))
        print("git hash of the figure: " + image_config.GIT_HASH)
        if str(cur_git_hash) == image_config.GIT_HASH:
            print(
                "You are on the correct git branch! Be wary of uncommitted changes in the repo at the time of image creation impacting figures."
            )
        else:
            print(
                "these git commit hashes dont line up, you may get errors or unexpected behavior, please checkout the commit of the figure"
            )
    return BasicMechanisticModel(**image_config.__dict__)
