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
            self.config.VAX_MODEL_DATA, vax_spline_filename
        )
        self.config.VAX_MODEL_DATA = vax_spline_path
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
        return self.inference_algo

    def likelihood(
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
            tf=max(obs_hosps_days) + 1,
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
        # numpyro.deterministic(
        #     "final_timestep_s", solution.ys[self.config.COMPARTMENT_IDX.S][-1]
        # )
        # numpyro.deterministic(
        #     "final_timestep_e", solution.ys[self.config.COMPARTMENT_IDX.E][-1]
        # )
        # numpyro.deterministic(
        #     "final_timestep_i", solution.ys[self.config.COMPARTMENT_IDX.I][-1]
        # )
        # numpyro.deterministic(
        #     "final_timestep_c", solution.ys[self.config.COMPARTMENT_IDX.C][-1]
        # )
        # sample intrinsic infection hospitalization rate here
        ihr_dict = {
            "ihr_0": self.config.ihr_0,
            "ihr_1": self.config.ihr_1,
            "ihr_2": self.config.ihr_2,
            "ihr_3": self.config.ihr_3,
            "ihr_immune_mult": self.config.ihr_immune_mult,
        }
        ihrs = self.sample_if_distribution(ihr_dict)
        ihr = jnp.array(
            [ihrs["ihr_0"], ihrs["ihr_1"], ihrs["ihr_2"], ihrs["ihr_3"]]
        )

        # sample ihr multiplier due to previous infection or vaccinations
        ihr_immune_mult = ihrs["ihr_immune_mult"]

        # sample ihr multiplier due to JN1 (assuming JN1 has less severity)
        ihr_jn1_mult = numpyro.sample("ihr_jn1_mult", Dist.Beta(100, 1))

        # calculate modelled hospitalizations based on the ihrs
        # add 1 to wane because we have time dimension prepended
        model_incidence = jnp.diff(
            solution.ys[self.config.COMPARTMENT_IDX.C],
            axis=0,
        )

        model_incidence_no_exposures_non_jn1 = jnp.sum(
            model_incidence[:, :, 0, 0, :2], axis=-1
        )
        model_incidence_no_exposures_jn1 = model_incidence[:, :, 0, 0, 2]
        model_incidence_all_non_jn1 = jnp.sum(
            model_incidence[:, :, :, :, :2], axis=(2, 3, 4)
        )
        model_incidence_all_jn1 = jnp.sum(
            model_incidence[:, :, :, :, 2], axis=(2, 3)
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
                jnp.bincount(hosps_interval_ind, m, length=len(obs_hosps_days))
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
        # never_infected = jnp.sum(
        #     solution.ys[self.config.COMPARTMENT_IDX.S][
        #         obs_sero_days, :, 0, :, :
        #     ],
        #     axis=(2, 3),
        # )
        # sim_seroprevalence = 1 - never_infected / self.config.POPULATION
        # sim_lseroprevalence = jnp.log(
        #     sim_seroprevalence / (1 - sim_seroprevalence)
        # )  # logit seroprevalence

        # mask = ~jnp.isnan(obs_sero_lmean)
        # with numpyro.handlers.mask(mask=mask):
        #     numpyro.sample(
        #         "lseroprevalence",
        #         Dist.Normal(sim_lseroprevalence, obs_sero_lsd),
        #         obs=obs_sero_lmean,
        #     )

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
