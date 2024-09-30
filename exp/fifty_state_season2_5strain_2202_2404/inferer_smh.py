import copy
import os

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as Dist
from jax.random import PRNGKey
from jax.typing import ArrayLike

import mechanistic_model.utils as utils
from mechanistic_model.mechanistic_inferer import MechanisticInferer


class SMHInferer(MechanisticInferer):
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

    def get_parameters(self):
        freeze_params = copy.deepcopy(self.config)
        parameters = {
            "INIT_DATE": freeze_params.INIT_DATE,
            "CONTACT_MATRIX": freeze_params.CONTACT_MATRIX,
            "POPULATION": freeze_params.POPULATION,
            "NUM_STRAINS": freeze_params.NUM_STRAINS,
            "NUM_AGE_GROUPS": freeze_params.NUM_AGE_GROUPS,
            "NUM_WANING_COMPARTMENTS": freeze_params.NUM_WANING_COMPARTMENTS,
            "WANING_PROTECTIONS": freeze_params.WANING_PROTECTIONS,
            "MAX_VACCINATION_COUNT": freeze_params.MAX_VACCINATION_COUNT,
            "STRAIN_INTERACTIONS": freeze_params.STRAIN_INTERACTIONS,
            "VACCINE_EFF_MATRIX": freeze_params.VACCINE_EFF_MATRIX,
            "BETA_TIMES": freeze_params.BETA_TIMES,
            "STRAIN_R0s": freeze_params.STRAIN_R0s,
            "INFECTIOUS_PERIOD": freeze_params.INFECTIOUS_PERIOD,
            "EXPOSED_TO_INFECTIOUS": freeze_params.EXPOSED_TO_INFECTIOUS,
            "INTRODUCTION_TIMES": freeze_params.INTRODUCTION_TIMES,
            "INTRODUCTION_SCALES": freeze_params.INTRODUCTION_SCALES,
            "INTRODUCTION_PCTS": freeze_params.INTRODUCTION_PCTS,
            "INITIAL_INFECTIONS_SCALE": freeze_params.INITIAL_INFECTIONS_SCALE,
            "CONSTANT_STEP_SIZE": freeze_params.CONSTANT_STEP_SIZE,
            "SEASONALITY_AMPLITUDE": freeze_params.SEASONALITY_AMPLITUDE,
            "SEASONALITY_SECOND_WAVE": freeze_params.SEASONALITY_SECOND_WAVE,
            "SEASONALITY_DOMINANT_WAVE_DAY": freeze_params.SEASONALITY_DOMINANT_WAVE_DAY,
            "SEASONALITY_SECOND_WAVE_DAY": freeze_params.SEASONALITY_SECOND_WAVE_DAY,
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
        # allows the ODEs to just pass time as a parameter, makes them look cleaner
        external_i_function_prefilled = jax.tree_util.Partial(
            self.external_i,
            introduction_times=parameters["INTRODUCTION_TIMES"],
            introduction_scales=parameters["INTRODUCTION_SCALES"],
            introduction_pcts=parameters["INTRODUCTION_PCTS"],
        )
        # # pre-calculate the minimum value of the seasonality curves
        seasonality_function_prefilled = jax.tree_util.Partial(
            self.seasonality,
            seasonality_amplitude=parameters["SEASONALITY_AMPLITUDE"],
            seasonality_second_wave=parameters["SEASONALITY_SECOND_WAVE"],
            dominant_wave_day=parameters["SEASONALITY_DOMINANT_WAVE_DAY"],
            second_wave_day=parameters["SEASONALITY_SECOND_WAVE_DAY"],
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
            }
        )

        parameters["STRAIN_R0s"] = jnp.array(
            [
                parameters["STRAIN_R0s"][0],
                parameters["STRAIN_R0s"][1],
                parameters["STRAIN_R0s"][2],
                parameters["STRAIN_R0s"][3],
                numpyro.deterministic(
                    "STRAIN_R0s_4", parameters["STRAIN_R0s"][2]
                ),
            ]
        )
        parameters["BETA"] = (
            parameters["STRAIN_R0s"] / parameters["INFECTIOUS_PERIOD"]
        )

        return parameters

    def seasonality(
        self,
        t: ArrayLike,
        seasonality_amplitude: ArrayLike,
        seasonality_second_wave: ArrayLike,
        dominant_wave_day: ArrayLike,
        second_wave_day: ArrayLike,
    ) -> ArrayLike:
        """A revamped seasonality function which allows for a dominant and second wave to vary in phase from one another
        As opposed to the original seasonality function which required both waves to be exactly 6 months apart
        the user may now specify two dates for the peaks of the seasonality, a dominant wave amplitude,
        and a coefficient on the second waves peak.
        Parameters
        ----------
        t : ArrayLike
            simulation day with t=0 being INIT_DATE
        seasonality_amplitude : ArrayLike
            amplitude size, with peak at 1+seasonality_amplitude and minimum at 1-seasonality_amplitude
        seasonality_second_wave : ArrayLike
            enforced 0 <= seasonality_second_wave <= 1.0
            adjusts how pronouced the second wave is,
            with 1.0 being equally sized to dominant wave, and 0 being no second wave
        dominant_wave_day : ArrayLike
            date on which dominant seasonality coefficient peak
        second_wave_day : ArrayLike
            date on which second seasonality coefficient peaks
        """

        def g(t: ArrayLike, phi: ArrayLike, w: int):
            return jnp.cos(2 * jnp.pi * (t - phi) / 730.0) ** w

        def f(
            t: ArrayLike,
            dominant_wave_day,
            second_wave_day,
            seasonality_second_wave: ArrayLike,
            w: int = 20,
        ):
            # w is a positive even integer used to enforce cos to be always positive
            # its magnitude increases the slope of the cosine curve, reaching its
            # peak faster and falling back to its minimum quicker
            return (
                -0.5
                + 1.0 * g(t, phi=dominant_wave_day, w=w)
                + seasonality_second_wave * g(t, phi=second_wave_day, w=w)
            )

        shifted_dominant_wave_day = (
            dominant_wave_day - self.config.INIT_DATE.timetuple().tm_yday
        )
        shifted_second_wave_day = (
            second_wave_day - self.config.INIT_DATE.timetuple().tm_yday
        )

        return (
            1
            + f(
                t,
                dominant_wave_day=shifted_dominant_wave_day,
                second_wave_day=shifted_second_wave_day,
                seasonality_second_wave=seasonality_second_wave,
            )
            * seasonality_amplitude
            * 2
        )

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
        if "INITIAL_INFECTIONS_SCALE" in parameters.keys():
            initial_state = self.scale_initial_infections(
                parameters["INITIAL_INFECTIONS_SCALE"]
            )
        else:
            initial_state = self.INITIAL_STATE

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
        # save the final timestep of solution array for each compartment
        numpyro.deterministic(
            "final_timestep_s", solution.ys[self.config.COMPARTMENT_IDX.S][-1]
        )
        numpyro.deterministic(
            "final_timestep_e", solution.ys[self.config.COMPARTMENT_IDX.E][-1]
        )
        numpyro.deterministic(
            "final_timestep_i", solution.ys[self.config.COMPARTMENT_IDX.I][-1]
        )
        numpyro.deterministic(
            "final_timestep_c", solution.ys[self.config.COMPARTMENT_IDX.C][-1]
        )
        # sample intrinsic infection hospitalization rate here
        # m_i is ratio btw the average across all states of the median of ihr_0 and the average across all states
        # of the median of ihr_3 produced from a previous fit
        # v_i is defined similarly, but as the variance
        # sample intrinsic infection hospitalization rate,
        # where the concentrations are based on the function fit_new_beta
        v_0, v_1, v_2 = (
            9.361000583593154e-05,
            6.948228259019488e-05,
            0.0002923281265483212,
        )

        m_0, m_1, m_2 = (
            0.020448716487218747,
            0.048698216511437936,
            0.1402618274806952,
        )

        ihr_mult_0 = numpyro.sample(
            "ihr_mult_0",
            Dist.Beta(
                (m_0 * (1 - m_0) / v_0 - 1) * m_0,
                (m_0 * (1 - m_0) / v_0 - 1) * (1 - m_0),
            ),
        )
        ihr_mult_1 = numpyro.sample(
            "ihr_mult_1",
            Dist.Beta(
                (m_1 * (1 - m_1) / v_1 - 1) * m_1,
                (m_1 * (1 - m_1) / v_1 - 1) * (1 - m_1),
            ),
        )
        ihr_mult_2 = numpyro.sample(
            "ihr_mult_2",
            Dist.Beta(
                (m_2 * (1 - m_2) / v_2 - 1) * m_2,
                (m_2 * (1 - m_2) / v_2 - 1) * (1 - m_2),
            ),
        )
        ihr_3 = numpyro.sample("ihr_3", Dist.Beta(60 * 20, 340 * 20))
        ihr = jnp.array(
            [ihr_3 * ihr_mult_0, ihr_3 * ihr_mult_1, ihr_3 * ihr_mult_2, ihr_3]
        )

        # sample ihr multiplier due to previous infection or vaccinations
        ihr_immune_mult = numpyro.sample(
            "ihr_immune_mult", Dist.Beta(100 * 6, 300 * 6)
        )

        # sample ihr multiplier due to JN1 (assuming JN1 has less severity)
        # ihr_jn1_mult = numpyro.sample("ihr_jn1_mult", Dist.Beta(100, 1))
        ihr_jn1_mult = numpyro.sample(
            "ihr_jn1_mult", Dist.Beta(380 * 3, 20 * 3)
        )

        # calculate modelled hospitalizations based on the ihrs
        # add 1 to wane because we have time dimension prepended
        model_incidence = jnp.diff(
            solution.ys[self.config.COMPARTMENT_IDX.C],
            axis=0,
        )
        model_incidence = jnp.sum(model_incidence, axis=4)

        model_incidence_no_exposures_non_jn1 = jnp.sum(
            model_incidence[:, :, 0, 0, :4], axis=-1
        )
        model_incidence_no_exposures_jn1 = model_incidence[:, :, 0, 0, 4]
        model_incidence_all_non_jn1 = jnp.sum(
            model_incidence[:, :, :, :, :4], axis=(2, 3, 4)
        )
        model_incidence_all_jn1 = jnp.sum(
            model_incidence[:, :, :, :, 4], axis=(2, 3)
        )
        model_incidence_w_exposures_non_jn1 = (
            model_incidence_all_non_jn1 - model_incidence_no_exposures_non_jn1
        )
        model_incidence_w_exposures_jn1 = (
            model_incidence_all_jn1 - model_incidence_no_exposures_jn1
        )

        # calculate weekly model hospitalizations with the two IHRs we created
        # TODO, should we average every 7 days or just pick every day from obs_metrics
        model_hosps = (
            model_incidence_no_exposures_non_jn1 * ihr
            + model_incidence_no_exposures_jn1 * ihr * ihr_jn1_mult
            + model_incidence_w_exposures_non_jn1 * ihr * ihr_immune_mult
            + model_incidence_w_exposures_jn1
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
            sim_seroprevalence = 1 - never_infected / self.config.POPULATION
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
                    self.config.I_AXIS_IDX.age + 1,
                    self.config.I_AXIS_IDX.hist + 1,
                    self.config.I_AXIS_IDX.vax + 1,
                    4,  # no current ENUM to fill this out lol
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
