"""
The following is a class which runs a series of ODE equations, performs inference, and returns Solution objects for analysis.
"""
import numpyro
from numpyro import distributions as Dist
from numpyro.infer import MCMC, NUTS
from functools import partial
import jax.numpy as jnp
import jax

numpyro.set_host_device_count(4)
jax.config.update("jax_enable_x64", True)


class mechanistic_runner:
    def __init__(self, initial_state, model, **kwargs):
        self.__dict__.update(kwargs)
        self.runtime_config = kwargs
        self.INITIAL_STATE = initial_state
        self.model = model

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
