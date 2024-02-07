"""
The following is a class which runs a series of ODE equations, performs inference, and returns Solution objects for analysis.
"""

import numpyro
from numpyro import distributions as Dist
from numpyro.infer import MCMC, NUTS
from diffrax import (
    ODETerm,
    PIDController,
    SaveAt,
    Solution,
    Tsit5,
    diffeqsolve,
)
from functools import partial
import jax.numpy as jnp
import jax
from jax.random import PRNGKey
import utils
import pandas as pd
import numpy as np
from jax.scipy.stats.norm import pdf
from config.config import Config
from enum import IntEnum

numpyro.set_host_device_count(4)
jax.config.update("jax_enable_x64", True)


class MechanisticRunner:

    def __init__(
        self, initial_state, model, runner_config_path, global_variables_path
    ):
        config = Config(global_variables_path).add_file(runner_config_path)
        # grab all parameters passed from global and initializer configs
        # TODO, move away from loading config into self
        self.__dict__.update(**config.__dict__)
        self.INITIAL_STATE = initial_state
        self.model = model
        # get population counts for each age bin
        self.retrieve_population_counts()
        self.load_cross_immunity_matrix()
        self.load_vaccination_model()
        self.load_external_i_distributions()
        self.load_contact_matrix()

    def get_args(
        self,
        sample: bool = False,
        sample_dist_dict: dict[str, Dist.Distribution] = {},
    ):
        """
        A function that returns model args as a dictionary as expected by the ODETerm function f(t, y(t), args)dt
        https://docs.kidger.site/diffrax/api/terms/#diffrax.ODETerm

        for example functions f() in charge of disease dynamics see the model_odes folder.

        Parameters
        ----------
        `sample`: boolean
            whether or not to sample key parameters, used when model is being run in MCMC and parameters are being infered
        `sample_dist_dict`: dict(str:numpyro.distribution)
            a dictionary of parameters to sample.
            follows format "parameter_name":numpyro.Distributions.dist(). DO NOT pass numpyro.sample() objects to the dictionary.

        Returns
        ----------
        dict{str: Object}: A dictionary where key value pairs are used as parameters by an ODE model
        """
        # get counts of the initial state compartments by age bin.
        # ignore the C compartment since it is just house keeping
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
                raise "Sample = True but not sample_dist_dict provided"
            # either using the default sample_dist_dict, or the one provided by the user
            # transform these distributions into numpyro samples.
            for key, item in sample_dist_dict.items():
                # if user wants to sample model initial infections, do that outside of get_args()
                if key == "INITIAL_INFECTIONS":
                    continue
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
        r0 = args.get("R0", self.STRAIN_R0s)
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

    def run(
        self,
        tf: int = 100,
        sample: bool = False,
        sample_dist_dict: dict[str, Dist.Distribution] = {},
    ):
        term = ODETerm(
            lambda t, state, parameters: self.model(state, t, parameters)
        )
        solver = Tsit5()
        t0 = 0.0
        dt0 = 1.0
        saveat = SaveAt(ts=jnp.linspace(t0, tf, int(tf) + 1))
        # if the user wants to sample model initial infections, do it here
        # initial_state = (
        #     self.load_initial_state(
        #         numpyro.sample(
        #             "INITIAL_INFECTIONS",
        #             sample_dist_dict["INITIAL_INFECTIONS"],
        #         )
        #     )
        #     if "INITIAL_INFECTIONS" in sample_dist_dict.keys()
        #     else self.INITIAL_STATE
        # )
        initial_state = self.INITIAL_STATE

        solution = diffeqsolve(
            term,
            solver,
            t0,
            tf,
            dt0,
            initial_state,
            args=self.get_args(
                sample=sample, sample_dist_dict=sample_dist_dict
            ),
            # discontinuities due to beta manipulation specified as jump_ts
            stepsize_controller=PIDController(
                rtol=1e-5,
                atol=1e-6,
                jump_ts=list(self.BETA_TIMES),
            ),
            saveat=saveat,
            # higher for large time scales / rapid changes
            max_steps=int(1e6),
        )
        self.solution = solution
        return solution

    def incidence(
        self,
        incidence: list[int],
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
            sample=True,
            sample_dist_dict=sample_dist_dict,
            tf=len(incidence),
        )
        # add 1 to idxs because we are straified by time in the solution object
        # sum down to just time x age bins
        model_incidence = jnp.sum(
            solution.ys[self.COMPARTMENT_IDX.C],
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

        Returns
        -----------
        numpyro.infer.MCMC object userd to sample parameters.
        This can be used to print summaries, pass along covariance matrices, or query posterier distributions
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
            negbin=negbin,
            sample_dist_dict=sample_dist_dict,
        )
        mcmc.print_summary()
        return mcmc

    @partial(jax.jit, static_argnums=(0))
    def external_i(self, t):
        """
        Given some time t, returns jnp.array of shape self.INITIAL_STATE[self.COMPARTMENT_IDX.I] representing external infected persons
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
            jnp.array(shape=(self.INITIAL_STATE[self.COMPARTMENT_IDX.I].shape)) of external individuals to the system
            interacting with susceptibles within the system, used to impact force of infection.
        """
        # set up our return value
        external_i_compartment = jnp.zeros(
            self.INITIAL_STATE[self.COMPARTMENT_IDX.I].shape
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
                    pdf,
                    loc=introduced_time_sampler,
                    scale=self.INTRODUCTION_SCALE,
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

    @partial(jax.jit, static_argnums=(0))
    def vaccination_rate(self, t):
        """
        Given some time t, returns a jnp.array of shape (self.NUM_AGE_GROUPS, self.MAX_VAX_COUNT + 1)
        representing the age / vax history stratified vaccination rates for an additional vaccine. Used by transmission models
        to determine vaccination rates at a particular time step.

        MUST BE CONTINUOUS AND DIFFERENTIABLE FOR ALL TIMES t. If you want a piecewise implementation of vax rates must declare jump points
        in the MCMC object.

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
            utils.evaluate_cubic_spline(
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

    def retrieve_population_counts(self):
        """
        A wrapper function which takes retrieves the age stratified population counts across all the INITIAL_STATE compartments
        (minus the book-keeping C compartment.)
        """
        self.POPULATION = np.sum(  # sum together S+E+I compartments
            np.array(
                [
                    np.sum(
                        compartment,
                        axis=(
                            self.S_AXIS_IDX.hist,
                            self.S_AXIS_IDX.vax,
                            self.S_AXIS_IDX.wane,
                        ),
                    )  # sum over all but age bin axis
                    for compartment in self.INITIAL_STATE[
                        : self.COMPARTMENT_IDX.C
                    ]  # avoid summing the book-keeping C compartment
                ]
            ),
            axis=(0),  # sum across compartments, keep age bins
        )

    def load_cross_immunity_matrix(self):
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
                pdf, loc=introduced_time, scale=self.INTRODUCTION_SCALE
            )

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
            self.DEMOGRAPHIC_DATA_PATH,
            self.REGIONS,
            self.NUM_AGE_GROUPS,
            self.MINIMUM_AGE,
            self.AGE_LIMITS,
        )["United States"]["avg_CM"]
