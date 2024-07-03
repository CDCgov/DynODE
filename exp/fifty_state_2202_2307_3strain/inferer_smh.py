import os

import jax.numpy as jnp
import numpyro
import numpyro.distributions as Dist
from jax.random import PRNGKey

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
        # TEMP SOLUTION FOR THIS EXPERIMENT, day 414 = April 1st 2022, this is used to initialize
        # the next epoch after we step back a little bit
        numpyro.deterministic(
            "april_1st_timestep_s",
            solution.ys[self.config.COMPARTMENT_IDX.S][414],
        )
        numpyro.deterministic(
            "april_1st_timestep_e",
            solution.ys[self.config.COMPARTMENT_IDX.E][414],
        )
        numpyro.deterministic(
            "april_1st_timestep_i",
            solution.ys[self.config.COMPARTMENT_IDX.I][414],
        )
        numpyro.deterministic(
            "april_1st_timestep_c",
            solution.ys[self.config.COMPARTMENT_IDX.C][414],
        )
        # sample intrinsic infection hospitalization rate here
        ihr_0 = numpyro.sample("ihr_0", Dist.Beta(1 * 10, 399 * 10))
        ihr_1 = numpyro.sample("ihr_1", Dist.Beta(4 * 4, 396 * 4))
        ihr_2 = numpyro.sample("ihr_2", Dist.Beta(6 * 4, 394 * 4))
        ihr_3 = numpyro.sample("ihr_3", Dist.Beta(40 * 4, 360 * 4))
        ihr = jnp.array([ihr_0, ihr_1, ihr_2, ihr_3])

        # sample ihr multiplier due to previous infection or vaccinations
        ihr_immune_mult = numpyro.sample(
            "ihr_immune_mult", Dist.Beta(100 * 10, 300 * 10)
        )

        # calculate modelled hospitalizations based on the ihrs
        # add 1 to wane because we have time dimension prepended
        model_incidence = jnp.sum(
            solution.ys[self.config.COMPARTMENT_IDX.C],
            axis=self.config.S_AXIS_IDX.wane + 1,
        )
        # pick out zero shots, 0 exposures
        model_incidence_no_exposures = jnp.diff(
            model_incidence[:, :, 0, 0], axis=0
        )

        model_incidence_prev_exposure = jnp.sum(model_incidence, axis=(2, 3))
        model_incidence_prev_exposure = jnp.diff(
            model_incidence_prev_exposure, axis=0
        )
        # subtract no exposures to get prev_exposures
        model_incidence_prev_exposure -= model_incidence_no_exposures

        # calculate weekly model hospitalizations with the two IHRs we created
        # TODO, should we average every 7 days or just pick every day from obs_metrics
        model_hosps = (
            model_incidence_no_exposures * ihr
            + model_incidence_prev_exposure * ihr * ihr_immune_mult
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
