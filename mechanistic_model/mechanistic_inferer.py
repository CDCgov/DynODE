import copy
import json
import os
import warnings

import jax.numpy as jnp
import numpy as np
import numpyro
from jax.random import PRNGKey
from numpyro import distributions as Dist
from numpyro.diagnostics import summary
from numpyro.infer import MCMC, NUTS

import utils
from config.config import Config
from mechanistic_model.abstract_parameters import AbstractParameters
from mechanistic_model.mechanistic_runner import MechanisticRunner


class MechanisticInferer(AbstractParameters):
    """
    A class responsible for managing the fitting process of a mechanistic runner.
    Taking in priors, sampling from their distributions, managing MCMC or the sampling/fitting proceedure of choice,
    and coordinating the parsing and use of the posterier distributions.
    """

    def __init__(
        self,
        global_variables_path: str,
        distributions_path: str,
        runner: MechanisticRunner,
        initial_state: tuple,
        prior_inferer: MCMC = None,
    ):
        distributions_json = open(distributions_path, "r").read()
        global_json = open(global_variables_path, "r").read()
        self.config = Config(global_json).add_file(distributions_json)
        self.runner = runner
        self.INITIAL_STATE = initial_state
        self.set_infer_algo(prior_inferer=prior_inferer)
        self.retrieve_population_counts()
        self.load_cross_immunity_matrix()
        self.load_vaccination_model()
        self.load_contact_matrix()

    def set_infer_algo(self, prior_inferer=None, inferer_type="mcmc"):
        """
        Sets the inferer's inference algorithm and sampler.
        If passed a previous inferer of the same inferer_type, uses posteriors to aid in the definition of new priors.
        This does require special configuration parameters to aid in transition between sequential inferers.

        Parameters
        ----------
        prior_inferer: None, numpyro.infer.MCMC
            the inferer algorithm of the previous sequential call to inferer.infer
            use posteriors in this previous call to help define the priors in the current call.
        """
        supported_infer_algos = ["mcmc"]
        if inferer_type.lower().strip() not in supported_infer_algos:
            raise NotImplementedError(
                "Inference algorithms other than MCMC have not been implemented yet,"
                "try overriding the method in a subclass to set self.inference_algo"
            )

        if inferer_type == "mcmc":
            # default to max tree depth of 5 if not specified
            tree_depth = getattr(self.config, "MAX_TREE_DEPTH", 5)
            self.inference_algo = MCMC(
                NUTS(
                    self.likelihood,
                    dense_mass=True,
                    max_tree_depth=tree_depth,
                    init_strategy=numpyro.infer.init_to_median,
                ),
                num_warmup=self.config.INFERENCE_NUM_WARMUP,
                num_samples=self.config.INFERENCE_NUM_SAMPLES,
                num_chains=self.config.INFERENCE_NUM_CHAINS,
                progress_bar=self.config.INFERENCE_PROGRESS_BAR,
            )
            if prior_inferer is not None:
                # may want to look into this here:
                # https://num.pyro.ai/en/stable/mcmc.html#id7
                assert isinstance(
                    prior_inferer, MCMC
                ), "the previous inferer is not of the same type."
                self.set_posteriors_if_exist(prior_inferer)

    def likelihood(self, obs_metrics):
        """
        Given some observed metrics, samples the likelihood of them occuring
        under a set of parameter distributions sampled by self.inference_algo.

        Currently expects hospitalization data and samples IHR using a negative binomial distribution.
        """
        solution = self.runner.run(
            self.INITIAL_STATE, args=self.get_parameters(), tf=len(obs_metrics)
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

        # sample infection hospitalization rate here
        with numpyro.plate("num_age", self.config.NUM_AGE_GROUPS):
            ihr = numpyro.sample("ihr", Dist.Beta(0.5, 10))

        numpyro.sample(
            "incidence",
            Dist.Poisson(model_incidence * ihr),
            obs=obs_metrics,
        )

    def set_posteriors_if_exist(self, prior_inferer: MCMC):
        """
        Given a `prior_inferer` object look at its samples, check to make sure that
        each parameter sampled has converging chains, then calculate the mean of
        each of the parameters samples, as well as the covariance between all of the parameter
        posterior distributions.

        To exclude certain chains from use in posteriors use `DROP_CHAINS`
        config argument as a list of chain indexes.

        To exclude certain parameters from use in posteriors because you are not
        sampling them this epoch, use the `DROP_POSTERIOR_PARAMETERS` config argument
        as a list of sample names as they appear in `prior_inferer.print_summary()`

        Parameters
        -----------
        prior_inferer: MCMC
            the inferer algorithm used in the previous epoch, or None.

        Updates
        -----------
        self.prior_inferer_particle_means : `np.ndarray`
            non-dropped parameter means across all non-dropped chains
        self.prior_inferer_particle_cov : `np.ndarray`
            non-dropped parameters covariance across all non-dropped chains
        self.prior_inferer_param_names : `list[str]`
            non-dropped parameter names
        self.cholesky_triangle_matrix : `jnp.ndarray`
            a cholesky bottom triangle matrix for each non-dropped parameter.
            Used in cholesky decomposition of a multivariate normal distribution

        Returns
        -----------
        None
        """
        if prior_inferer is not None:
            # get all the samples from each chain run in previous inference
            samples = prior_inferer.get_samples(group_by_chain=True)
            # if a user does not want to use posteriors for certain parameters
            # they can drop them using the DROP_POSTERIOR_PARAMETERS keyword
            for parameter in getattr(
                self.config, "DROP_POSTERIOR_PARAMETERS", []
            ):
                samples.pop(parameter, None)
            # flatten any parameters that are created via numpyro.plate
            # these parameters add a dimensions to `samples` values, and mess with things
            samples = utils.flatten_list_parameters(samples)
            dropped_chains = []
            if hasattr(self.config, "DROP_CHAINS"):
                dropped_chains = self.config.DROP_CHAINS
            # if user specified they want certain chain indexes dropped, do that
            samples = utils.drop_sample_chains(samples, dropped_chains)
            # create a summary of these chains to calculate divergence of chains etc
            sample_summaries = summary(samples)
            # do some sort of testing to ensure the chains are properly converging.
            # lets all flatten all the samples from all chains together
            samples_array_flattened = None
            for sample in samples.keys():
                sample_summary = sample_summaries[sample]
                divergent_chains = False
                if sample_summary["r_hat"] > 1.05:
                    warnings.warn(
                        "WARNING: the inferer has detected divergent chains in the %s parameter "
                        "being passed as input into this epoch. "
                        "Diverging chains can cause summary posterior distributions to not "
                        "accurately reflect the true posterior distribution "
                        "you may use the DROP_CHAINS configuration parameter to "
                        "drop the offending chain " % str(sample),
                        RuntimeWarning,
                    )
                    divergent_chains = True
                # now we add the parameter in flattened form for later
                if samples_array_flattened is None:
                    samples_array_flattened = [samples[sample].flatten()]
                else:
                    samples_array_flattened = np.concatenate(
                        (samples_array_flattened, [samples[sample].flatten()]),
                        axis=0,
                    )
            # if we have divergent chains, warn and show them to the user
            if divergent_chains:
                utils.plot_sample_chains(samples)
            # samples_array_flattened is now of shape (P, N*M)
            # for P parameters, N samples per chain and M chains per parameter
            self.prior_inferer_particle_means = np.mean(
                samples_array_flattened, axis=1
            )
            self.prior_inferer_particle_cov = np.cov(samples_array_flattened)
            self.prior_inferer_param_names = list(samples.keys())
            self.cholesky_triangle_matrix = jnp.linalg.cholesky(
                self.prior_inferer_particle_cov
            )

        return None

    def sample_if_distribution(self, parameters):
        """
        given a dictionary of keys and parameters, searches through all keys
        and samples the distribution associated with that key, if it exists.
        Otherwise keeps the value associated with that key.
        Converts lists with distributions inside to `jnp.ndarray`

        Parameters
        ----------
        parameters: dict{str: obj}
            a dictionary mapping a parameter name to an object, either a value or a distribution.
            `numpyro.distribution` objects are sampled, and their sampled value replaces the distribution object
            within parameters. Capable of sampling lists with static values and distributions together.

        Returns
        ----------
        parameters dictionary with any `numpyro.distribution` objects replaced with jax.tracer samples
        of those distributions from `numpyro.sample`
        """
        for key, param in parameters.items():
            # if distribution, sample and replace
            if issubclass(type(param), Dist.Distribution):
                param = numpyro.sample(key, param)
            # if list, check for distributions within and replace them
            elif isinstance(param, (np.ndarray, list)) and any(
                [
                    issubclass(type(param_lst), Dist.Distribution)
                    for param_lst in param
                ]
            ):
                param = jnp.array(
                    [
                        (
                            numpyro.sample(key + "_" + str(i), param_lst)
                            if issubclass(type(param_lst), Dist.Distribution)
                            else param_lst
                        )
                        for i, param_lst in enumerate(param)
                    ]
                )
            # else static param, do nothing
            parameters[key] = param
        return parameters

    def get_parameters(self):
        """
        Goes through the parameters passed to the inferer, if they are distributions, it samples them.
        Otherwise it returns their raw values.

        Converts all list types with sampled values to jax tracers.

        Returns
        -----------
        dict{str:obj} where obj may either be a float value,
        or a jax tracer, in the case of a sampled value or list containing sampled values.
        """
        # multiple chains of MCMC calling get_parameters() should not share references, deep copy
        freeze_params = copy.deepcopy(self.config)
        parameters = {
            "CONTACT_MATRIX": freeze_params.CONTACT_MATRIX,
            "POPULATION": freeze_params.POPULATION,
            "NUM_STRAINS": freeze_params.NUM_STRAINS,
            "NUM_AGE_GROUPS": freeze_params.NUM_AGE_GROUPS,
            "NUM_WANING_COMPARTMENTS": freeze_params.NUM_WANING_COMPARTMENTS,
            "WANING_PROTECTIONS": freeze_params.WANING_PROTECTIONS,
            "MAX_VAX_COUNT": freeze_params.MAX_VAX_COUNT,
            "CROSSIMMUNITY_MATRIX": freeze_params.CROSSIMMUNITY_MATRIX,
            "VAX_EFF_MATRIX": freeze_params.VAX_EFF_MATRIX,
            "BETA_TIMES": freeze_params.BETA_TIMES,
            "STRAIN_R0s": freeze_params.STRAIN_R0s,
            "INFECTIOUS_PERIOD": freeze_params.INFECTIOUS_PERIOD,
            "EXPOSED_TO_INFECTIOUS": freeze_params.EXPOSED_TO_INFECTIOUS,
            "INTRODUCTION_TIMES": freeze_params.INTRODUCTION_TIMES,
        }
        parameters = self.sample_if_distribution(parameters)
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
        # add final parameters, if your model expects added parameters, add them here
        parameters = dict(
            parameters,
            **{
                "BETA": beta,
                "SIGMA": sigma,
                "GAMMA": gamma,
                "WANING_RATES": waning_rates,
                "EXTERNAL_I": self.external_i,
                "VACCINATION_RATES": self.vaccination_rate,
                "BETA_COEF": self.beta_coef,
            }
        )

        return parameters

    def infer(self, obs_metrics):
        """
        Infer parameters given priors inside of self.config, returns an inference_algo object with posterior distributions for each sampled parameter.
        Parameters
        -----------
        obs_metrics: jnp.array
            observed metrics on which likelihood will be calculated on to tune parameters.

        Returns
        -----------
        an inference object, often numpyro.infer.MCMC object used to infer parameters.
        This can be used to print summaries, pass along covariance matrices, or query posterier distributions
        """
        self.inference_algo.run(
            rng_key=PRNGKey(self.config.INFERENCE_PRNGKEY),
            obs_metrics=obs_metrics,
        )
        self.inference_algo.print_summary()
        return self.inference_algo

    def checkpoint(self, checkpoint_folder):
        """
        a function which saves the posterior samples from `self.inference_algo.get_samples()` into `checkpoint_folder` as well
        as the average final state of the model when using all particles found in `self.inference_algo.get_samples()`.
        """
        if self.inference_algo.last_state is None:
            print(
                "unable to checkpoint as you have not called infer() yet to produce posteriors"
            )
            return
        # get posterior samples
        samples = self.inference_algo.get_samples(group_by_chain=False)
        # we cant convert ndarray to samples, so we convert to list first
        for parameter in samples.keys():
            param_samples = samples[parameter]
            if isinstance(param_samples, (np.ndarray, jnp.ndarray)):
                samples[parameter] = param_samples.tolist()
        with open(os.path.join(checkpoint_folder, "sample.json"), "w") as file:
            json.dump(samples, file)

    def create_average_final_state(self, tf):
        if self.inference_algo.last_state is None:
            print(
                "unable to checkpoint as you have not called infer() yet to produce posteriors"
            )
            return
        samples = self.inference_algo.get_samples(group_by_chain=True)
        # lets flatten any parameters that come as lists into separate parameters
        samples = utils.flatten_list_parameters(samples)
        parameters_sampled = samples.keys()
        # now lets resize it to get rid of the chains dimension
        # current shape is (# params, num chains, num samples per chain)
        samples = np.array(samples.values())
        particles = samples.reshape(
            (samples.shape[0], samples.shape[1] * samples.shape[2])
        )  # (num_parameters, num_samples)
        num_particles = particles.shape[1]
        config_with_distributions = copy.deepcopy(self.config)
        all_final_states = [
            [],
            [],
            [],
            [],
        ]
        # for each particle, change self.config parameters to match the posterior parameter values
        # where applicable, sometimes we sample things like IHR which are not used by the runner
        for particle in range(num_particles):
            for i, parameter in enumerate(parameters_sampled):
                posterior_param_value = samples[i, particle]
                # if parameter name matches what is in self.config
                if hasattr(self.config, parameter):
                    setattr(self.config, parameter, posterior_param_value)
                # if we sample as a part of a list, the end of `parameter` will have _i where i is the index
                elif hasattr(self.config, "".join(parameter.split("_")[:-1])):
                    # remove the _i from the name so we can reference it in self.config
                    param_name_no_index = "".join(parameter.split("_")[:-1])
                    # grab the index so we can set that element of the list
                    param_index = int(parameter[-1])
                    self.config.__dict__[param_name_no_index][
                        param_index
                    ] = posterior_param_value
            # with self.config changed out, there are no distributions to sample, so calling self.get_parameters()
            # returns only values and no numpyro.sample tracers
            run_with_posterior_params = self.runner.run(
                self.INITIAL_STATE,
                self.get_parameters(),
                tf=tf,
            )
            for i, compartment in enumerate(run_with_posterior_params.ys):
                # timestep is first dimension, so we selecting compartment at t=-1
                # which in this case is t=240
                last_time_step = compartment[-1]
                # add a rolling sum to the average_final_state variable
                all_final_states[i].append(last_time_step)
                # total_pops[i] += np.sum(last_time_step)
        average_final_state = [
            np.mean(compartment, axis=0) for compartment in all_final_states
        ]
        # set our self.config back to what it was in case we use inferer again
        self.config = config_with_distributions
        return average_final_state
