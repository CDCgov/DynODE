"""
The following code is used to fit a series of prior parameter distributions via running them
through Ordinary Differential Equations (ODEs) and comparing the likelihood of the output to some
observed metrics.
"""

import json
import warnings
from typing import Union

import jax.numpy as jnp
import jax.typing
import numpy as np
import numpyro  # type: ignore
from diffrax import Solution
from jax.random import PRNGKey
from numpyro import distributions as Dist
from numpyro.diagnostics import summary  # type: ignore
from numpyro.infer import MCMC, NUTS  # type: ignore

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
        initial_state: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
        prior_inferer: MCMC = None,
    ):
        distributions_json = open(distributions_path, "r").read()
        global_json = open(global_variables_path, "r").read()
        self.config = Config(global_json).add_file(distributions_json)
        self.runner = runner
        self.INITIAL_STATE = initial_state
        self.infer_complete = False  # flag once inference completes
        self.set_infer_algo(prior_inferer=prior_inferer)
        self.retrieve_population_counts()
        self.load_vaccination_model()
        self.load_contact_matrix()

    def set_infer_algo(
        self, prior_inferer: MCMC = None, inferer_type: str = "mcmc"
    ) -> None:
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

    def likelihood(
        self, obs_metrics: Union[jax.Array, None] = None, tf: int = 100
    ) -> Solution:
        """
        Given some observed metrics, samples the likelihood of them occuring
        under a set of parameter distributions sampled by self.inference_algo.

        if obs_metrics is None likelihood will not actually fit to values and instead return Solutions
        based on randomly sampled values.

        if obs_metrics is None, will run model for runs for `tf` days
        otherwise runs for `len(obs_metrics)` days

        Currently expects hospitalization data and samples IHR.
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
            tf=len(obs_metrics) if obs_metrics is not None else tf,
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
        poisson_rates = jnp.maximum(model_incidence, 1e-6)

        # save the final timestep of solution array for each compartment
        # this is useful for checkpointing model epochs
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
        # sample infection hospitalization rate here
        with numpyro.plate("num_age", self.config.NUM_AGE_GROUPS):
            ihr = numpyro.sample("ihr", Dist.Beta(0.5, 10))

        numpyro.sample(
            "incidence",
            Dist.Poisson(poisson_rates * ihr),
            obs=obs_metrics,
        )
        return solution

    def set_posteriors_if_exist(self, prior_inferer: MCMC) -> None:
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
            samples_array_flattened = np.array([])
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
                if not len(samples_array_flattened):
                    samples_array_flattened = np.array(
                        [samples[sample].flatten()]
                    )
                else:
                    samples_array_flattened = np.concatenate(
                        (
                            samples_array_flattened,
                            np.array([samples[sample].flatten()]),
                        ),
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

    def infer(self, obs_metrics: jax.typing.ArrayLike) -> MCMC:
        """
        Infer parameters given priors inside of self.config,
        returns an inference_algo object with posterior distributions for each sampled parameter.
        Parameters
        -----------
        obs_metrics: jnp.array
            observed metrics on which likelihood will be calculated on to tune parameters.
            See `likelihood()` method for implemented definition of `obs_metrics`

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
        self.inference_timesteps = len(obs_metrics)
        self.infer_complete = True
        return self.inference_algo

    def checkpoint(
        self, checkpoint_path: str, group_by_chain: bool = True
    ) -> None:
        """
        a function which saves the posterior samples from `self.inference_algo` into `checkpoint_path` as a json file.
        will save anything sampled or numpyro.deterministic as long as it is tracked by `self.inference_algo`.



        Parameters
        -----------
        checkpoint_path: str
            a path to which the json file is saved to. Throws error if folders do not exist, overwrites existing JSON files within.


        Raises
        ----------
        ValueError
            if inference has not been called (not self.infer_complete),
            and thus there are no posteriors to be saved to `checkpoint_path`


        Returns
        -----------
        None
        """
        if not self.infer_complete:
            raise ValueError(
                "unable to checkpoint as you have not called infer() yet to produce posteriors"
            )
        # get posterior samples including any calls to numpyro.deterministic
        if group_by_chain:
            samples = self.inference_algo._states[
                self.inference_algo._sample_field
            ]
        else:
            samples = self.inference_algo._states_flat[
                self.inference_algo._sample_field
            ]
        # we cant convert ndarray to samples, so we convert to list first
        for parameter in samples.keys():
            param_samples = samples[parameter]
            if isinstance(param_samples, (np.ndarray, jnp.ndarray)):
                samples[parameter] = param_samples.tolist()
        with open(checkpoint_path, "w") as file:
            json.dump(samples, file)

    def load_posterior_particle(
        self,
        particle_num: int = 0,
        randomize: bool = False,
        tf: Union[int, None] = None,
        external_posteriors: dict[str, jax.Array] = {},
    ) -> list[Solution]:
        """
        if self.infer_complete loads a particle across chains of sampled posteriors,
        either at random if `randomize`, or with an index `particle_num`.

        if `external_posteriors` are specified, uses them instead of self.inference_algo.get_samples()
        to load particles.

        Returns a dictionary mapping (particle, chain) combinations to the diffrax.Solution object produced
        by running self.likelihood with that posterior particle stored under "solution" and the
        posterior particle values themselves under "posteriors"

        if `randomize=True` and random particle_num will be selected, values passed to particle_num will be ignored.

        Parameters
        ------------
        particle_num: int
            what particle to run posteriors for across all chains
        randomize: bool
            flag whether or not to pick a random particle
        tf: Union[int, None]:
            number of days to run posterior model for, defaults to same number of days used in fitting.
        external_posteriors: dict
            if you want to use posteriors defined somewhere outside of this instance of the MechanisticInferer.
            For example, if you saved posteriors from an Azure Batch job to a file and want to reload them,
            then a dictionary containing a copy of that inferers inference_algo.get_samples(group_by_chain=True)
            may be used here to apply those posteriors onto this model

        Returns
        ---------------
        `dict[tuple(int, int):dict[str: Solution, str: dict[str:jnp.ndarray]]]` a dictionary containing
        the solution timeline of the model run with the posteriors at (particle_num, chain_num)
        with values at `dict[(particle_num, chain_num)]["posteriors"]` and timeline found at
        `dict[(particle_num, chain_num)]["solution"]`

        Example
        --------------
        <insert 2 chain inference above>
        load_posterior_particle(particle_num=100) = {(100, 0): {solution: diffrax.Solution, "posteriors": {...}},
                                                     (100, 1): {solution: diffrax.Solution, "posteriors": {...}} ...}

        Note
        ------------
        Very important note if you choose to use `external_posteriors`. In the scenario
        this instance of `MechanisticInferer.likelihood` samples parameters not named in `external_posteriors`
        they will be RESAMPLED AT RANDOM. This method will not error and will instead fill in those
        missing samples according to the PRNGKey seeded with self.config.INFERENCE_PRNGKEY.

        This may be useful to you if you wish to obtain confidence intervals by varying a particular value.
        """
        if not self.infer_complete and not external_posteriors:
            raise RuntimeError(
                "Attempting to load a posterior particle before fitting to any data, "
                "run self.infer() first to produce posterior particles"
            )
        if tf is None:
            # run for same amount of timesteps as given in inference
            # given to exists since self.infer_complete is True
            if hasattr(self, "inference_timesteps"):
                tf = self.inference_timesteps
            # unless user is using external_posterior, we may have not inferred yet
            else:
                raise RuntimeError(
                    "You are using external_posteriors to load posterios but did not provide `tf`, "
                    "this instance does not have access to fitting data to know how long to run for"
                )
        # first step, get the posterior particle number
        if randomize:
            particle_num = int(
                np.random.RandomState(self.config.INFERENCE_PRNGKEY).uniform(
                    low=0, high=self.config.INFERENCE_NUM_SAMPLES
                )
            )
        if not external_posteriors:
            # Get posterior samples
            posterior_samples = self.inference_algo.get_samples(
                group_by_chain=True
            )
        else:
            # user defined their own posteriors they want to run, use those
            posterior_samples = external_posteriors

        return self._load_posterior_single_chain(
            posterior_samples, particle_num, tf
        )

    def _load_posterior_single_chain(
        self,
        posterior_samples: dict[str, jax.Array],
        particle_num: int,
        tf: int,
    ):
        """
        PRIVATE FUNCTION
        used by `load_posterior_particle` to colate each posterior particle
        run on `self.likelihood` across chains
        Dont touch unless you know what you are doing.
        """
        # filter down to just chosen particle across chains
        # re-evaluate number of chains since we may be using external_posteriors run in diff context
        single_chain = {}
        num_chains = self.config.INFERENCE_NUM_CHAINS
        for key in posterior_samples.keys():
            num_chains = posterior_samples[key].shape[0]
            single_chain[key] = posterior_samples[key][:, particle_num]

        # for each chain run that individual particle through the model, save Solution
        solutions_by_chain = {}
        # Substitute sampled values back into the model
        for chain in range(num_chains):
            # pull out the sampled values of this particular chain
            single_particle = {}
            for key in posterior_samples.keys():
                single_particle[key] = single_chain[key][chain]

            solutions_by_chain[
                (particle_num, chain)
            ] = self._load_posterior_single_particle(single_particle, tf)
        return solutions_by_chain

    def _load_posterior_single_particle(
        self, single_particle: dict[str, jax.Array], tf: int
    ):
        """
        PRIVATE FUNCTION
        used by `load_posterior_particle` to actually execute a single posterior particle on `self.likelihood`
        Dont touch unless you know what you are doing.
        """
        # run the model with the same seed, but this time
        # all calls to numpyro.sample() will lookup the value from single_particle_chain
        # instead! this is effectively running the particle
        substituted_model = numpyro.handlers.substitute(
            numpyro.handlers.seed(
                self.likelihood,
                jax.random.PRNGKey(self.config.INFERENCE_PRNGKEY),
            ),
            single_particle,
        )
        sol = substituted_model(tf=tf)
        return {
            "solution": sol,
            "posteriors": substituted_model.data,
        }
