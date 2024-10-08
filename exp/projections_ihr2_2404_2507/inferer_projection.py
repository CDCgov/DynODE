import copy
import os
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as Dist
from jax.random import PRNGKey
from jax.scipy.stats.norm import pdf
from numpyro.infer import MCMC

import mechanistic_model.utils as utils
from config.config import Config
from mechanistic_model import SEIC_Compartments
from mechanistic_model.mechanistic_inferer import MechanisticInferer
from mechanistic_model.mechanistic_runner import MechanisticRunner


class ProjectionParameters(MechanisticInferer):

    def __init__(
        self,
        global_variables_path: str,
        distributions_path: str,
        runner: MechanisticRunner,
        prior_inferer: MCMC = None,
    ):
        """A specialized init method which does not take an initial state, this is because
        posterior particles will contain the initial state used."""
        distributions_json = open(distributions_path, "r").read()
        global_json = open(global_variables_path, "r").read()
        self.config = Config(global_json).add_file(distributions_json)
        self.runner = runner
        self.infer_complete = False  # flag once inference completes
        self.set_infer_algo(prior_inferer=prior_inferer)
        self.load_vaccination_model()
        self.load_contact_matrix()

    def load_vaccination_model(self):
        """
        an overridden version of the vaccine model so we can load
        state-specific vaccination splines using the REGIONS parameter
        """
        vax_spline_filename = "spline_fits_%s.csv" % (
            self.config.REGIONS[0].lower().replace(" ", "_")
        )
        vax_spline_path = os.path.join(
            self.config.VACCINATION_MODEL_DATA, vax_spline_filename
        )
        self.config.VACCINATION_MODEL_DATA = vax_spline_path
        super().load_vaccination_model()

    def infer(
        self,
        obs_hosps,
        obs_hosps_days,
        obs_sero_lmean,
        obs_sero_lsd,
        obs_sero_days,
        obs_var_prop,
        obs_var_days,
        obs_var_sd,
    ):
        """
        OVERRIDEN TO ADD MORE DATA STREAMS TO COMPARE AGAINST
        Infer parameters given priors inside of self.config, returns an inference_algo object with posterior distributions for each sampled parameter.


        Parameters
        ----------
        obs_hosps: jnp.ndarray: weekly hosp incidence values from NHSN
        obs_hosps_days: list[int] the sim day on which each obs_hosps value is measured.
                        for example obs_hosps[0] = 0 = self.config.INIT_DATE
        obs_sero_lmean: jnp.ndarray: observed seroprevalence in logit scale
        obs_sero_lsd: jnp.ndarray: standard deviation of logit seroprevalence (use this to
                      control the magnitude of uncertainty / weightage of fitting)
        obs_sero_days: list[int] the sim day on which each obs_sero value is measured.
                       e.g., [9, 23, ...] meaning that we have data on day 9, 23, ...

        Returns
        -----------
        an inference object, often numpyro.infer.MCMC object used to infer parameters.
        This can be used to print summaries, pass along covariance matrices, or query posterier distributions
        """
        self.inference_algo.run(
            rng_key=PRNGKey(self.config.INFERENCE_PRNGKEY),
            obs_hosps=obs_hosps,
            obs_hosps_days=obs_hosps_days,
            obs_sero_lmean=obs_sero_lmean,
            obs_sero_lsd=obs_sero_lsd,
            obs_sero_days=obs_sero_days,
            obs_var_prop=obs_var_prop,
            obs_var_days=obs_var_days,
            obs_var_sd=obs_var_sd,
        )
        self.inference_algo.print_summary()
        self.infer_complete = True
        self.inference_timesteps = max(obs_hosps_days) + 1
        return self.inference_algo

    def sample_strain_x_intro_time(self, offset):
        """
        Samples a value of the strain X intro time based on the lags between introduction times of the fitted strains
        """
        # use numpyro.sample to read in the posterior values of the intro times
        # we only introduced strains BA2BA5 - KP, so go through those (index exclusive so ends at X)
        past_introduction_times = jnp.array(
            [
                numpyro.sample(
                    "INTRODUCTION_TIMES_%s" % idx,
                    numpyro.distributions.Normal(),
                )
                for idx, _ in enumerate(
                    self.config.STRAIN_IDX._member_names_[
                        self.config.STRAIN_IDX.BA2BA5 : self.config.STRAIN_IDX.KP
                    ]
                )
            ]
        )
        # get the day of year of intro for XBB1 and JN1 (fall variants)
        init_yday = 42  # TODO: this is hardcoded
        xbb1_yday = (init_yday + past_introduction_times[1]) % 365
        jn1_yday = (init_yday + past_introduction_times[3]) % 365
        mean_yday = (xbb1_yday + jn1_yday) / 2
        sd_yday = 14  # fix at 14 days sd
        yday_dist = Dist.Normal(loc=mean_yday, scale=sd_yday)

        strain_x_intro_yday = numpyro.sample("INTRO_YDAY_X", yday_dist)
        # ensure that intro_yday is non-negative
        strain_x_intro_yday = jnp.max(jnp.array([0, strain_x_intro_yday]))
        # if yday smaller than offset, add 365 to it
        strain_x_intro_time_raw = strain_x_intro_yday - offset
        strain_x_intro_time_raw = jnp.where(
            strain_x_intro_time_raw < 0,
            365 + strain_x_intro_time_raw,
            strain_x_intro_time_raw,
        )

        # max with zero to avoid negatives
        strain_x_intro_time = numpyro.deterministic(
            "INTRODUCTION_TIME_X", strain_x_intro_time_raw
        )
        return strain_x_intro_time

    @partial(jax.jit, static_argnums=(0))
    def vaccination_rate(self, t):
        vaccine_offset = getattr(self.config, "ZERO_VACCINE_DAY", 0.0)
        vaccine_rate_mult = getattr(
            self.config, "VACCINATION_RATE_MULTIPLIER", 1.0
        )
        vaccine_rate_mult = jnp.where(
            t < vaccine_offset, 0.0, vaccine_rate_mult
        )
        t_offset = jnp.where(t < vaccine_offset, 0.0, t - vaccine_offset)
        return vaccine_rate_mult * super().vaccination_rate(t_offset)

    def get_parameters(self):
        """
        Overriding the get_parameters() method to work with an undefined initial state
        because projections only define their initial state by sampling the `final_timestep` parameter
        we need self.POPULATION to only be evaluated after self.INITIAL_STATE has been pulled from the posteriors

        this method is close to super().get_parameters() but difers because it does not use `self.POPULATION` since
        it does not exist yet, it must be set only AFTER initial state exists.
        """

        freeze_params = copy.deepcopy(self.config)
        # copied code vebaitem from super().get_parameters()
        parameters = {
            "INIT_DATE": freeze_params.INIT_DATE,
            "CONTACT_MATRIX": freeze_params.CONTACT_MATRIX,
            "NUM_STRAINS": freeze_params.NUM_STRAINS,
            # "POPULATION": freeze_params.POPULATION,
            "NUM_AGE_GROUPS": freeze_params.NUM_AGE_GROUPS,
            "NUM_WANING_COMPARTMENTS": freeze_params.NUM_WANING_COMPARTMENTS,
            "WANING_PROTECTIONS": freeze_params.WANING_PROTECTIONS,
            "MAX_VACCINATION_COUNT": freeze_params.MAX_VACCINATION_COUNT,
            "STRAIN_INTERACTIONS": freeze_params.STRAIN_INTERACTIONS,
            "VACCINE_EFF_MATRIX": freeze_params.VACCINE_EFF_MATRIX,
            "BETA_TIMES": freeze_params.BETA_TIMES,
            "STRAIN_R0s": freeze_params.STRAIN_R0s,
            "R0_MULTIPLIER": freeze_params.R0_MULTIPLIER,
            "INFECTIOUS_PERIOD": freeze_params.INFECTIOUS_PERIOD,
            "EXPOSED_TO_INFECTIOUS": freeze_params.EXPOSED_TO_INFECTIOUS,
            "INTRODUCTION_TIMES": freeze_params.INTRODUCTION_TIMES,
            "INTRODUCTION_SCALES": freeze_params.INTRODUCTION_SCALES,
            "INTRODUCTION_PCTS": freeze_params.INTRODUCTION_PCTS,
            "INITIAL_INFECTIONS_SCALE": freeze_params.INITIAL_INFECTIONS_SCALE,
            "CONSTANT_STEP_SIZE": freeze_params.CONSTANT_STEP_SIZE,
            "SEASONALITY_AMPLITUDE": freeze_params.SEASONALITY_AMPLITUDE,
            "SEASONALITY_SECOND_WAVE": freeze_params.SEASONALITY_SECOND_WAVE,
            "SEASONALITY_SHIFT": freeze_params.SEASONALITY_SHIFT,
            "MIN_HOMOLOGOUS_IMMUNITY": freeze_params.MIN_HOMOLOGOUS_IMMUNITY,
        }
        parameters = utils.sample_if_distribution(parameters)
        # re-create the CROSSIMMUNITY_MATRIX since we may be sampling the STRAIN_INTERACTIONS matrix now
        parameters["CROSSIMMUNITY_MATRIX"] = (
            utils.strain_interaction_to_cross_immunity2(
                freeze_params.NUM_STRAINS, parameters["STRAIN_INTERACTIONS"]
            )
        )
        # create parameters based on other possibly sampled parameters
        beta = parameters["STRAIN_R0s"] / parameters["INFECTIOUS_PERIOD"]
        gamma = 1 / parameters["INFECTIOUS_PERIOD"]
        sigma = 1 / parameters["EXPOSED_TO_INFECTIOUS"]
        # last waning time is zero since last compartment does not wane
        # catch a division by zero error here.
        waning_rates = np.array(
            [
                1 / waning_time if waning_time > 0 else 0
                for waning_time in freeze_params.WANING_TIMES
            ]
        )

        # numpyro needs to believe it is sampling from something in order for the override to work
        def fake_sampler():
            return numpyro.distributions.Normal()

        # fake_sampler will be overriden in runtime by the `final_timestep` values of the posteriors
        # final_timestep refers to the final state of the system after fitting, aka day 0 of projection
        parameters["INITIAL_STATE"] = tuple(
            [
                jnp.array(numpyro.sample("final_timestep_s", fake_sampler())),
                jnp.array(numpyro.sample("final_timestep_e", fake_sampler())),
                jnp.array(numpyro.sample("final_timestep_i", fake_sampler())),
                jnp.array(numpyro.sample("final_timestep_c", fake_sampler())),
            ]
        )
        # inserting this line here, needs to be in freeze_params to avoid memory leaks
        # of multiple chains modifying self.POPULATION
        freeze_params.POPULATION = self.retrieve_population_counts(
            parameters["INITIAL_STATE"]
        )
        # if we are sampling the KP's R0 multiplier
        if self.config.SAMPLE_KP_R0_MULTIPLIER:
            r0_multiplier_dist = Dist.Uniform(0.88, 0.97)
            r0_multiplier = numpyro.sample(
                "KP_R0_MULTIPLIER", r0_multiplier_dist
            )
            parameters["R0_MULTIPLIER"] = r0_multiplier
        # if we are introducing a strain, the INTRODUCTION_TIMES array will be non-empty
        # and if we specifically want to sample strain_x intro time, we set that flag to True
        if (
            parameters["INTRODUCTION_TIMES"][1]
            and self.config.SAMPLE_STRAIN_X_INTRO_TIME
        ):
            init_yday = parameters["INIT_DATE"].timetuple().tm_yday
            strain_x_intro_time = self.sample_strain_x_intro_time(
                offset=init_yday
            )
            # reset our intro time to the sampled lag distribution intro time for strain X
            parameters["INTRODUCTION_TIMES"][1] = strain_x_intro_time
        # allows the ODEs to just pass time as a parameter, makes them look cleaner
        external_i_function_prefilled = jax.tree_util.Partial(
            self.external_i,
            introduction_times=parameters["INTRODUCTION_TIMES"],
            introduction_scales=parameters["INTRODUCTION_SCALES"],
            introduction_pcts=parameters["INTRODUCTION_PCTS"],
            population=freeze_params.POPULATION,
        )
        # # pre-calculate the minimum value of the seasonality curves
        seasonality_function_prefilled = jax.tree_util.Partial(
            self.seasonality,
            seasonality_amplitude=parameters["SEASONALITY_AMPLITUDE"],
            seasonality_second_wave=parameters["SEASONALITY_SECOND_WAVE"],
            seasonality_shift=parameters["SEASONALITY_SHIFT"],
        )
        # add final parameters, if your model expects added parameters, add them here
        parameters = dict(
            parameters,
            **{
                "BETA": beta,
                "SIGMA": sigma,
                "GAMMA": gamma,
                "WANING_RATES": waning_rates,
                "EXTERNAL_I": external_i_function_prefilled,
                "VACCINATION_RATES": self.vaccination_rate,
                "BETA_COEF": self.beta_coef,
                "SEASONAL_VACCINATION_RESET": self.seasonal_vaccination_reset,
                "SEASONALITY": seasonality_function_prefilled,
                "POPULATION": freeze_params.POPULATION,
            }
        )
        # new code for projections in particular
        parameters["STRAIN_R0s"] = jnp.array(
            [
                parameters["STRAIN_R0s"][0],
                parameters["STRAIN_R0s"][1],
                parameters["STRAIN_R0s"][2],
                parameters["STRAIN_R0s"][3],
                numpyro.deterministic(
                    "STRAIN_R0s_4", parameters["STRAIN_R0s"][2]
                ),
                parameters["R0_MULTIPLIER"]
                * numpyro.deterministic(
                    "STRAIN_R0s_5", parameters["STRAIN_R0s"][3]
                ),
                numpyro.deterministic(
                    "STRAIN_R0s_6", parameters["STRAIN_R0s"][2]
                ),
            ]
        )
        parameters["BETA"] = (
            parameters["STRAIN_R0s"] / parameters["INFECTIOUS_PERIOD"]
        )
        avg_2strain_interaction = (
            parameters["STRAIN_INTERACTIONS"][5, 3]
            + parameters["STRAIN_INTERACTIONS"][4, 2]
        ) / 2
        avg_1strain_interaction = (
            parameters["STRAIN_INTERACTIONS"][5, 4]
            + parameters["STRAIN_INTERACTIONS"][4, 3]
        ) / 2
        immune_escape_64 = (1 - avg_2strain_interaction) * parameters[
            "STRAIN_INTERACTIONS"
        ][6, 4]
        immune_escape_65 = (1 - avg_1strain_interaction) * parameters[
            "STRAIN_INTERACTIONS"
        ][6, 5]
        parameters["STRAIN_INTERACTIONS"] = (
            parameters["STRAIN_INTERACTIONS"]
            .at[6, 4]
            .set(1 - immune_escape_64)
        )
        parameters["STRAIN_INTERACTIONS"] = (
            parameters["STRAIN_INTERACTIONS"]
            .at[6, 5]
            .set(1 - immune_escape_65)
        )
        parameters["CROSSIMMUNITY_MATRIX"] = (
            utils.strain_interaction_to_cross_immunity2(
                self.config.NUM_STRAINS, parameters["STRAIN_INTERACTIONS"]
            )
        )

        return parameters

    def retrieve_population_counts(self, initial_state):
        return np.sum(  # sum together S+E+I compartments
            np.array(
                [
                    np.sum(
                        compartment,
                        axis=(
                            self.config.S_AXIS_IDX.hist,
                            self.config.S_AXIS_IDX.vax,
                            self.config.S_AXIS_IDX.wane,
                        ),
                    )  # sum over all but age bin axis
                    for compartment in initial_state[
                        : self.config.COMPARTMENT_IDX.C
                    ]  # avoid summing the book-keeping C compartment
                ]
            ),
            axis=(0),  # sum across compartments, keep age bins
        )

    def rework_initial_state(self, initial_state):
        """
        Take the original `initial_state` which is (4, 7, 3, 4) -> (4, 8, 4, 4) and add an
        additional strain to the infection history and an additional vax tier and the infected by dimensions for E+I
        """
        s_new = jnp.pad(
            initial_state[0], [(0, 0), (0, 2), (0, 1), (0, 0)], mode="constant"
        )
        e_new = jnp.pad(
            initial_state[1], [(0, 0), (0, 2), (0, 1), (0, 2)], mode="constant"
        )
        i_new = jnp.pad(
            initial_state[2], [(0, 0), (0, 2), (0, 1), (0, 2)], mode="constant"
        )
        c_new_shape = list(s_new.shape)
        c_new_shape.append(i_new.shape[3])
        c_new = jnp.zeros(tuple(c_new_shape))
        initial_state = (
            s_new,
            e_new,
            i_new,
            c_new,
        )
        return initial_state

    def scale_initial_infections(
        self, scale_factor, INITIAL_STATE
    ) -> SEIC_Compartments:
        """
        overriden version that does not use self.INITIAL_STATE
        a function which modifies returns a modified version of
        self.INITIAL_STATE scaling the number of initial infections by `scale_factor`.

        Preserves the ratio of the Exposed/Infectious compartment population sizes.
        Does not modified self.INITIAL_STATE, returns a copy.

        Parameters
        ----------
        scale_factor: float
            a multiplier value >=0.0.
            `scale_factor` < 1 reduces number of initial infections,
            `scale_factor` == 1.0 leaves initial infections unchanged,
            `scale_factor` > 1 increases number of initial infections.

        Returns
        ---------
        A copy of INITIAL_INFECTIONS with each compartment being scaled according to `scale_factor`
        """
        pop_counts_by_compartment = jnp.array(
            [
                jnp.sum(compartment)
                for compartment in INITIAL_STATE[
                    : self.config.COMPARTMENT_IDX.C
                ]
            ]
        )
        initial_infections = (
            pop_counts_by_compartment[self.config.COMPARTMENT_IDX.E]
            + pop_counts_by_compartment[self.config.COMPARTMENT_IDX.I]
        )
        initial_susceptibles = pop_counts_by_compartment[
            self.config.COMPARTMENT_IDX.S
        ]
        # total_pop_size = initial_susceptibles + initial_infections
        new_infections_size = scale_factor * initial_infections
        # negative if scale_factor < 1.0
        gained_infections = new_infections_size - initial_infections
        scale_factor_susceptible_compartment = 1 - (
            gained_infections / initial_susceptibles
        )
        # multiplying E and I by the same scale_factor preserves their relative ratio
        scale_factors = [
            scale_factor_susceptible_compartment,
            scale_factor,
            scale_factor,
            1.0,  # for the C compartment, unchanged.
        ]
        # scale each compartment and return
        initial_state = tuple(
            [
                compartment * factor
                for compartment, factor in zip(INITIAL_STATE, scale_factors)
            ]
        )
        return initial_state

    @partial(jax.jit, static_argnums=(0))
    def external_i(
        self,
        t,
        introduction_times: jax.Array,
        introduction_scales: jax.Array,
        introduction_pcts: jax.Array,
        population,
    ) -> jax.Array:
        """
        Given some time t, returns jnp.array of shape self.INITIAL_STATE[self.config.COMPARTMENT_IDX.I] representing external infected persons
        interacting with the population. it does so by calling some function f_s(t) for each strain s.

        MUST BE CONTINUOUS AND DIFFERENTIABLE FOR ALL TIMES t.

        The stratafication of the external population is decided by the introduced strains, which are defined by
        3 parallel lists of the time they peak (`introduction_times`),
        the number of external infected individuals introduced as a % of the tracked population (`introduction_pcts`)
        and how quickly or slowly those individuals contact the tracked population (`introduction_scales`)

        Parameters
        ----------
        `t`: float as Traced<ShapedArray(float64[])>
            current time in the model, due to the just-in-time nature of Jax this float value may be contained within a
            traced array of shape () and size 1. Thus no explicit comparison should be done on "t".

        `introduction_times`: list[int] as Traced<ShapedArray(float64[])>
            a list representing the times at which external strains should be introduced, in days, after t=0 of the model
            This list is ordered inversely to self.config.STRAIN_R0s. If 2 external strains are defined, the two
            values in `introduction_times` will refer to the last 2 STRAIN_R0s, not the first two.

        `introduction_scales`: list[float] as Traced<ShapedArray(float64[])>
            a list representing the standard deviation of the curve that external strains are introduced with, in days
            This list is ordered inversely to self.config.STRAIN_R0s. If 2 external strains are defined, the two
            values in `introduction_times` will refer to the last 2 STRAIN_R0s, not the first two.

        `introduction_pcts`: list[float] as Traced<ShapedArray(float64[])>
            a list representing the proportion of each age bin in self.POPULATION[self.config.INTRODUCTION_AGE_MASK]
            that will be exposed to the introduced strain over the entire course of the introduction.
            This list is ordered inversely to self.config.STRAIN_R0s. If 2 external strains are defined, the two
            values in `introduction_times` will refer to the last 2 STRAIN_R0s, not the first two.

        Returns
        -----------
        external_i_compartment: jax.Array
            jnp.array(shape=(self.INITIAL_STATE[self.config.COMPARTMENT_IDX.I].shape)) of external individuals to the system
            interacting with susceptibles within the system, used to impact force of infection.
        """

        # define a function that returns 0 for non-introduced strains
        def zero_function(_):
            return 0

        external_i_distributions = [
            zero_function for _ in range(self.config.NUM_STRAINS)
        ]
        introduction_percentage_by_strain = [0] * self.config.NUM_STRAINS
        for introduced_strain_idx, (
            introduced_time,
            introduction_scale,
            introduction_perc,
        ) in enumerate(
            zip(introduction_times, introduction_scales, introduction_pcts)
        ):
            # earlier introduced strains earlier will be placed closer to historical strains (0 and 1)
            dist_idx = (
                self.config.NUM_STRAINS
                - self.config.NUM_INTRODUCED_STRAINS
                + introduced_strain_idx
            )
            # use a normal PDF with std dv
            external_i_distributions[dist_idx] = partial(
                pdf, loc=introduced_time, scale=introduction_scale
            )
            introduction_percentage_by_strain[dist_idx] = introduction_perc
        # with our external_i_distributions set up, now we can execute them on `t`
        # set up our return value
        external_i_compartment = jnp.zeros(
            (
                self.config.NUM_AGE_GROUPS,
                self.config.NUM_STRAINS + 1,
                self.config.MAX_VACCINATION_COUNT + 1,
                self.config.NUM_STRAINS,
            )
        )
        introduction_age_mask = jnp.where(
            jnp.array(self.config.INTRODUCTION_AGE_MASK),
            1,
            0,
        )
        for strain in self.config.STRAIN_IDX:
            external_i_distribution = external_i_distributions[strain]
            introduction_perc = introduction_percentage_by_strain[strain]
            external_i_compartment = external_i_compartment.at[
                introduction_age_mask, 0, 0, strain
            ].set(
                external_i_distribution(t)
                * introduction_perc
                * population[introduction_age_mask]
            )
        return external_i_compartment

    def likelihood(
        self,
        obs_hosps=None,
        obs_hosps_days=None,
        obs_sero_lmean=None,
        obs_sero_lsd=None,
        obs_sero_days=None,
        obs_var_prop=None,
        obs_var_days=None,
        obs_var_sd=None,
        tf=None,
        infer_mode=True,
    ):
        """
        overridden likelihood that takes as input weekly hosp data starting from self.config.INIT_DATE

        Parameters
        ----------
        obs_hosps: jnp.ndarray: weekly hosp incidence values from NHSN
        obs_hosps_days: list[int] the sim day on which each obs_hosps value is measured.
                        for example obs_hosps[0] = 0 = self.config.INIT_DATE
        obs_sero_lmean: jnp.ndarray: observed seroprevalence in logit scale
        obs_sero_lsd: jnp.ndarray: standard deviation of logit seroprevalence (use this to
                      control the magnitude of uncertainty / weightage of fitting)
        obs_sero_days: list[int] the sim day on which each obs_sero value is measured.
                       e.g., [9, 23, ...] meaning that we have data on day 9, 23, ...
        """
        parameters = self.get_parameters()
        print(parameters["STRAIN_R0s"][5])
        initial_state = self.rework_initial_state(parameters["INITIAL_STATE"])

        solution = self.runner.run(
            initial_state,
            args=parameters,
            tf=max(obs_hosps_days) + 1 if tf is None else tf,
        )
        # add 1 to idxs because we are stratified by time in the solution object
        # sum down to just time x age bins
        model_incidence = jnp.sum(
            solution.ys[self.config.COMPARTMENT_IDX.C],
            axis=(
                self.config.I_AXIS_IDX.hist + 1,
                self.config.I_AXIS_IDX.vax + 1,
                self.config.I_AXIS_IDX.strain + 1,
            ),
        )
        # axis = 0 because we take diff across time
        model_incidence = jnp.diff(model_incidence, axis=0)
        # sample intrinsic infection hospitalization rate here
        ihr_mult_prior_means = jnp.array([0.02, 0.05, 0.14])
        ihr_mult_prior_variances = (
            jnp.array(
                [
                    9.36e-05,
                    6.94e-05,
                    0.00029,
                ]
            )
            / 4
        )

        ihr_mult_prior_a = (
            (
                ihr_mult_prior_means
                * (1 - ihr_mult_prior_means)
                / ihr_mult_prior_variances
            )
            - 1
        ) * ihr_mult_prior_means
        ihr_mult_prior_b = (
            (
                ihr_mult_prior_means
                * (1 - ihr_mult_prior_means)
                / ihr_mult_prior_variances
            )
            - 1
        ) * (1 - ihr_mult_prior_means)

        ihr_mult_0 = numpyro.sample(
            "ihr_mult_0", Dist.Beta(ihr_mult_prior_a[0], ihr_mult_prior_b[0])
        )
        ihr_mult_1 = numpyro.sample(
            "ihr_mult_1", Dist.Beta(ihr_mult_prior_a[1], ihr_mult_prior_b[1])
        )
        ihr_mult_2 = numpyro.sample(
            "ihr_mult_2", Dist.Beta(ihr_mult_prior_a[2], ihr_mult_prior_b[2])
        )
        ihr_3 = numpyro.sample("ihr_3", Dist.Beta(40 * 10, 360 * 10))
        ihr = jnp.array([ihr_mult_0, ihr_mult_1, ihr_mult_2, 1]) * ihr_3

        # sample ihr multiplier due to previous infection or vaccinations
        ihr_immune_mult = numpyro.sample(
            "ihr_immune_mult", Dist.Beta(100 * 6, 300 * 6)
        )

        # sample ihr multiplier due to JN1 (assuming JN1 has less severity)
        # ihr_jn1_mult = numpyro.sample(
        #     "ihr_jn1_mult", Dist.Beta(400 * 4, 4 * 4)
        # )
        ihr_jn1_mult = numpyro.deterministic("ihr_jn1_mult", 0.95)

        # calculate modelled hospitalizations based on the ihrs
        # add 1 to wane because we have time dimension prepended
        model_incidence = jnp.diff(
            solution.ys[self.config.COMPARTMENT_IDX.C],
            axis=0,
        )

        model_incidence_no_exposures_non_jn1 = jnp.sum(
            model_incidence[:, :, 0, 0, :, :4], axis=(-1, -2)
        )
        model_incidence_no_exposures_jn1 = jnp.sum(
            model_incidence[:, :, 0, 0, :, 4:], axis=(-1, -2)
        )
        model_incidence_wbooster_non_jn1 = jnp.sum(
            model_incidence[:, :, :, 3, :, :4], axis=(-1, -2, -3)
        )
        model_incidence_wbooster_jn1 = jnp.sum(
            model_incidence[:, :, :, 3, :, 4:], axis=(-1, -2, -3)
        )
        model_incidence_all_non_jn1 = jnp.sum(
            model_incidence[:, :, :, :, :, :4], axis=(2, 3, 4, 5)
        )
        model_incidence_all_jn1 = jnp.sum(
            model_incidence[:, :, :, :, :, 4:], axis=(2, 3, 4, 5)
        )
        model_incidence_wexp_no_booster_non_jn1 = (
            model_incidence_all_non_jn1
            - model_incidence_no_exposures_non_jn1
            - model_incidence_wbooster_non_jn1
        )
        model_incidence_wexp_no_booster_jn1 = (
            model_incidence_all_jn1
            - model_incidence_no_exposures_jn1
            - model_incidence_wbooster_jn1
        )

        # calculate weekly model hospitalizations with the two IHRs we created
        # TODO, should we average every 7 days or just pick every day from obs_metrics
        booster_ihr_reduction = getattr(
            self.config, "BOOSTER_IHR_REDUCTION", 0.0
        )
        model_hosps = (
            model_incidence_no_exposures_non_jn1 * ihr
            + model_incidence_no_exposures_jn1 * ihr * ihr_jn1_mult
            + model_incidence_wbooster_non_jn1
            * ihr
            * ihr_immune_mult
            * (1 - booster_ihr_reduction)
            + model_incidence_wbooster_jn1
            * ihr
            * ihr_immune_mult
            * ihr_jn1_mult
            * (1 - booster_ihr_reduction)
            + model_incidence_wexp_no_booster_non_jn1 * ihr * ihr_immune_mult
            + model_incidence_wexp_no_booster_jn1
            * ihr
            * ihr_immune_mult
            * ihr_jn1_mult
        )

        if infer_mode:
            # obs_hosps_days = [6, 13, 20, ....]
            # Incidence from day 0, 1, 2, ..., 6 goes to first bin, day 7 - 13 goes to second bin...
            # break model_hosps into chunks of intervals and aggregate them
            # first, find out which interval goes to which days
            hosps_interval_ind = jnp.searchsorted(
                jnp.array(obs_hosps_days), jnp.arange(max(obs_hosps_days) + 1)
            )
            # for observed, multiply number by number of days within an interval
            obs_hosps_interval = (
                obs_hosps
                * jnp.bincount(hosps_interval_ind, length=len(obs_hosps_days))[
                    :, None
                ]
            )
            # for simulated, aggregate by index
            sim_hosps_interval = jnp.array(
                [
                    jnp.bincount(
                        hosps_interval_ind, m, length=len(obs_hosps_days)
                    )
                    for m in model_hosps.T
                ]
            ).T
            # x.shape = [650, 4]
            # for x[0:7, :] -> y[0, :]
            # y.shape = [65, 4]
            mask_incidence = ~jnp.isnan(obs_hosps_interval)
            with numpyro.handlers.mask(mask=mask_incidence):
                numpyro.sample(
                    "incidence",
                    Dist.Poisson(sim_hosps_interval),
                    obs=obs_hosps_interval,
                )

            ## Seroprevalence
            never_infected = jnp.sum(
                solution.ys[self.config.COMPARTMENT_IDX.S][
                    obs_sero_days, :, 0, :, :
                ],
                axis=(2, 3),
            )
            sim_seroprevalence = 1 - never_infected / parameters["POPULATION"]
            sim_lseroprevalence = jnp.log(
                sim_seroprevalence / (1 - sim_seroprevalence)
            )  # logit seroprevalence

            mask_sero = ~jnp.isnan(obs_sero_lmean)
            with numpyro.handlers.mask(mask=mask_sero):
                numpyro.sample(
                    "lseroprevalence",
                    Dist.Normal(sim_lseroprevalence, obs_sero_lsd),
                    obs=obs_sero_lmean,
                )

            ## Variant proportion
            strain_incidence = jnp.sum(
                solution.ys[self.config.COMPARTMENT_IDX.C],
                axis=(
                    self.config.C_AXIS_IDX.age + 1,
                    self.config.C_AXIS_IDX.hist + 1,
                    self.config.C_AXIS_IDX.vax + 1,
                    self.config.C_AXIS_IDX.wane + 1,
                ),
            )
            strain_incidence = jnp.diff(strain_incidence, axis=0)[
                : (max(obs_var_days) + 1)
            ]
            var_interval_ind = jnp.searchsorted(
                jnp.array(obs_var_days), jnp.arange(max(obs_var_days) + 1)
            )
            strain_incidence_interval = jnp.array(
                [
                    jnp.bincount(var_interval_ind, m, length=len(obs_var_days))
                    for m in strain_incidence.T
                ]
            ).T
            sim_var_prop = jnp.array(
                [incd / jnp.sum(incd) for incd in strain_incidence_interval]
            )
            sim_var_sd = jnp.ones(sim_var_prop.shape) * obs_var_sd

            numpyro.sample(
                "variant_proportion",
                Dist.Normal(sim_var_prop, sim_var_sd),
                obs=obs_var_prop,
            )
        return {
            "solution": solution,
            "hospitalizations": model_hosps,
            "parameters": parameters,
        }
