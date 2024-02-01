"""
The following is a class which runs a series of ODE equations, performs inference, and returns Solution objects for analysis.
"""
import numpyro
from numpyro import distributions as Dist
from numpyro.infer import MCMC, NUTS
from functools import partial
import jax.numpy as jnp
import jax
import utils
import pandas as pd
from jax.scipy.stats.norm import pdf

numpyro.set_host_device_count(4)
jax.config.update("jax_enable_x64", True)


class MechanisticRunner:
    def __init__(self, initial_state, model, **kwargs):
        self.__dict__.update(kwargs)
        self.runtime_config = kwargs
        self.INITIAL_STATE = initial_state
        self.model = model
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
        TODO change this  TODO if sample=True, and no sample_dist_dict supplied, infectious period and BA1.1 introduction time are automatically sampled.

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
        pass

    def run(
        tf: int = 100,
        sample: bool = False,
        sample_dist_dict: dict[str, Dist.Distribution] = {},
    ):
        pass

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
        pass

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

    @partial(jax.jit, static_argnums=(0))
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
        pass
        # return jnp.exp(
        #     utils.VAX_FUNCTION(
        #         t + self.DAYS_AFTER_INIT_DATE,
        #         self.VAX_MODEL_KNOT_LOCATIONS,
        #         self.VAX_MODEL_BASE_EQUATIONS,
        #         self.VAX_MODEL_KNOTS,
        #     )
        # )

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
            self.DEMOGRAPHIC_DATA,
            self.REGIONS,
            self.NUM_AGE_GROUPS,
            self.MINIMUM_AGE,
            self.AGE_LIMITS,
        )["United States"]["avg_CM"]
