"""
The following code is used to fit a series of prior parameter distributions via running them
through Ordinary Differential Equations (ODEs) and comparing the likelihood of the output to some
observed metrics.
"""

import json
from typing import Union

import bayeux as bx
import jax.numpy as jnp
import jax.typing
import numpy as np
import numpyro  # type: ignore
from diffrax import Solution  # type: ignore
from jax.random import PRNGKey
from numpyro import distributions as Dist
from numpyro.infer import MCMC, NUTS  # type: ignore

from . import SEIC_Compartments
from .abstract_parameters import AbstractParameters
from .config import Config
from .mechanistic_runner import MechanisticRunner


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
        initial_state: SEIC_Compartments,
    ):
        distributions_json = open(distributions_path, "r").read()
        global_json = open(global_variables_path, "r").read()
        self.config = Config(global_json).add_file(distributions_json)
        self.runner = runner
        self.INITIAL_STATE = initial_state
        self.infer_complete = False
        self.set_infer_algo()
        self.retrieve_population_counts()
        self.load_vaccination_model()
        self.load_contact_matrix()

    def set_infer_algo(self, inferer_type: str = "mcmc") -> None:
        """Sets the inferer's inference algorithm and sampler.

        Parameters
        ----------
        inferer_type : str, optional
            infer algo you wish to use, by default "mcmc"

        Raises
        ------
        NotImplementedError
            if passed `inferer_type` that is not yet supported, raises
            NotImplementedError
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

    def _get_predictions(
        self, parameters: dict, solution: Solution
    ) -> jax.Array:
        """generates post-hoc predictions from solved timeseries in `Solution` and
        parameters used to generate them within `parameters`. This will often be hospitalizations
        but could be more than just that.

        Parameters
        ----------
        parameters : dict
            parameters object returned by `get_parameters()` possibly containing information about the
            infection hospitalization ratio
        solution : Solution
            Solution object returned by `_solve_runner` or any call to `self.runner.run()`
            containing compartment timeseries

        Returns
        -------
        jax.Array or tuple[jax.Array]
            one or more jax arrays representing the different post-hoc predictions generated from
            `solution`. If fitting upon hospitalizations only, then a single jax.Array representing hospitalizations will be present.
        """
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
        # override this function for more complicated hospitalization logic
        with numpyro.plate("num_age", self.config.NUM_AGE_GROUPS):
            ihr = numpyro.sample("ihr", Dist.Beta(0.5, 10))
        hospitalizations = model_incidence * ihr
        return hospitalizations

    def run_simulation(
        self, tf: int
    ) -> dict[str, Union[Solution, jax.Array],]:
        parameters = self.get_parameters()
        solution = self._solve_runner(parameters, tf, self.runner)
        hospitalizations = self._get_predictions(parameters, solution)
        return {
            "solution": solution,
            "hospitalizations": hospitalizations,
            "parameters": parameters,
        }

    def likelihood(
        self,
        tf: int,
        obs_metrics: jax.Array,
    ):
        """
        Given some observed metrics, samples the likelihood of them occuring
        under a set of parameter distributions sampled by self.inference_algo.
        If `obs_metrics` is not defined and `infer_mode=False`, returns a dictionary
        containing the Solution object returned by `self.runner`, the hospitalizations
        predicted by the model, and the parameters returned by `self.get_parameters()`

        if obs_metrics is None likelihood will not actually fit to values and instead return Solutions
        based on randomly sampled values.

        if obs_metrics is None, will run model for runs for `tf` days
        otherwise runs for `len(obs_metrics)` days. If both `tf` and `obs_metrics` are None, raises RuntimeError.

        Currently expects hospitalization data and samples IHR.

        Parameters
        ----------
        obs_metrics : jax.Array, optional
            observed data, currently expecting hospitalization data, by default None
        tf : int, optional
            days to run model for, if obs_metrics is not provided, this parameter is used, by default None
        infer_mode : bool, optional
            whether or not to sample log likelihood of hospitalizations
            using `obs_metrics` as observed variables, by default True

        Returns
        -------
        dict[str, Union[Solution, jax.Array, dict]]
            dictionary containing three keys, `solution`, `hospitalizations`, and `parameters`
            containing the `Solution` object returned by self.runner, the predicted hospitalizations, and
            the parameters run respectively

        Raises
        ------
        RuntimeError
            if obs_metrics is None AND tf is none, raises runtime error. Need one or the other
        """
        dct = self.run_simulation(tf)
        solution = dct["solution"]
        predicted_metrics = dct["hospitalizations"]
        assert isinstance(solution, Solution)
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
        predicted_metrics = jnp.maximum(predicted_metrics, 1e-6)
        numpyro.sample(
            "incidence",
            Dist.Poisson(predicted_metrics),
            obs=obs_metrics,
        )

    def infer(self, obs_metrics: jax.Array) -> MCMC:
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
            tf=len(obs_metrics),
        )
        self.inference_algo.print_summary()
        self.inference_timesteps = len(obs_metrics)
        self.infer_complete = True
        return self.inference_algo

    def _debug_likelihood(self, **kwargs) -> bx.Model:
        """uses Bayeux to recreate the self.likelihood function for purposes of basic sanity checking

        passes all parameters given to it to `self.likelihood`, initializes with `self.INITIAL_STATE`
        and passes `self.config.INFERENCE_PRNGKEY` as seed for randomness.

        Returns
        -------
        Bayeux.Model
            model object used to debug
        """
        bx_model = bx.Model.from_numpyro(
            jax.tree_util.Partial(self.likelihood, **kwargs),
            # this does not work for non-one/sampled self.INITIAL_INFECTIONS_SCALE
            initial_state=self.INITIAL_STATE,
        )
        bx_model.mcmc.numpyro_nuts.debug(
            seed=PRNGKey(self.config.INFERENCE_PRNGKEY)
        )
        return bx_model

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
        particles: Union[tuple[int, int], list[tuple[int, int]]],
        tf: Union[int, None] = None,
        external_particle: dict[str, jax.Array] = {},
        verbose: bool = False,
    ) -> dict[
        tuple[int, int],
        dict[str, Union[Solution, jax.Array, dict[str, jax.Array]]],
    ]:
        """
        loads a list (or singular) of particles defined by a chain/particle tuple.
        Using sampled values from self.inference_algo.get_samples() to run
        `self.likelihood` with static values from that particle.

        if `external_posteriors` are specified, uses them instead of self.inference_algo.get_samples()
        to load static particle values.

        Parameters
        ------------
        particles: Union[tuple[int, int], list[tuple[int, int]]]
            a single tuple or list of tuples, each of which specifies the (chain_num, particle_num) to load
            will error if values are out of range of what was sampled.
        tf: Union[int, None]:
            number of days to run posterior model for, defaults to same number of days used in fitting
            if possible.
        external_posteriors: dict
            for use of particles defined somewhere outside of this instance of the MechanisticInferer.
            For example, loading a checkpoint.json containing saved posteriors from a different run.
            expects keys that match those given to `numpyro.sample` often from
            inference_algo.get_samples(group_by_chain=True).
        verbose: bool, optional
            whether or not to pring out the current chain_particle value being executed


        Returns
        ---------------
        `dict[tuple(int, int)]` a dictionary containing
        the returned value of `self.likelihood` evaluated with values from (chain_num, particle_num).
        Posterior values used append to the dictionary under the "posteriors" key.

        Example
        --------------
        <insert 2 chain inference above>
        `load_posterior_particle([(0, 100), [1, 120],...]) = {(0, 100): {solution: diffrax.Solution, "posteriors": {...}},
                                                     (1, 120): {solution: diffrax.Solution, "posteriors": {...}} ...}`

        Note
        ------------
        Very important note if you choose to use `external_posteriors`. In the scenario
        this instance of `MechanisticInferer.likelihood` samples parameters not named in `external_posteriors`
        they will be RESAMPLED according to the distribution passed in the config.
        This method will also salt the RNG key used on the prior according to the
        chain & particule numbers currently being run.


        This may be useful to you if you wish to fit upon some data, then introduce
        a new varying parameter over the posteriors (often during projection).
        """
        # if its a single particle, convert to len(1) list for simplicity
        if isinstance(particles, tuple):
            particles = [particles]
        if not self.infer_complete and not external_particle:
            raise RuntimeError(
                "Attempting to load a posterior particle before fitting to any data, "
                "run self.infer() first to produce posterior particles or pass externally produced particles"
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
        if not external_particle:
            # Get posterior samples
            posterior_samples = self.inference_algo.get_samples(
                group_by_chain=True
            )
        else:
            # user defined their own particle they want to run, use those
            posterior_samples = external_particle

        return_dct = {}
        for particle in particles:
            # get the particle chain and number
            chain_num, particle_num = particle
            if verbose:
                print(
                    "Executing (chain, particle): (%s, %s)"
                    % (str(chain_num), str(particle_num))
                )
            single_particle_samples = {}
            # go through each posterior and select that specific chain and particle
            for param in posterior_samples.keys():
                single_particle_samples[param] = posterior_samples[param][
                    chain_num
                ][particle_num]
            # run likelihood of that particular chain and particle
            single_particle_dct = self._load_posterior_single_particle(
                single_particle_samples,
                tf,
                chain_paricle_seed=chain_num * 10000 + particle_num,
            )
            # add this particle/chain run onto the return dict
            return_dct[(chain_num, particle_num)] = single_particle_dct
        return return_dct

    def _load_posterior_single_particle(
        self,
        single_particle: dict[str, jax.typing.ArrayLike],
        tf: int,
        chain_paricle_seed: int,
    ) -> dict:
        """
        PRIVATE FUNCTION
        used by `load_posterior_particle` to actually execute a single posterior particle on `self.likelihood`
        Dont touch unless you know what you are doing.

        Parameters
        ----------
        single_particle : dict[str, jax.typing.ArrayLike]
            a dictionary linking a parameter name to its posterior value,
            a single value or list depending on the sampled parameter
        tf : int
            the number of days to run the posteriors for
        chain_paricle_seed : int
            some salting unique to the particle being run, used to randomize any NEW parameters sampled that are
            not within `single_particle`

        Returns
        -------
        dict[str: [jax.Array, Solution]]
            a solution_dict containing the return value of `self.likelihood` as well as
            a field `posteriors` containing the values within `single_particle` as well as
            any new sampled values created by `self.likelihood` that were not found in `single_particle`
        """
        # run the model with the same seed, but this time
        # all calls to numpyro.sample() will lookup the value from single_particle_chain
        # instead! this is effectively running the particle
        substituted_model = numpyro.handlers.substitute(
            numpyro.handlers.seed(
                self.run_simulation,
                jax.random.PRNGKey(
                    self.config.INFERENCE_PRNGKEY + chain_paricle_seed
                ),
            ),
            single_particle,
        )
        sol_dct = substituted_model(tf=tf)
        sol_dct["posteriors"] = substituted_model.data
        return sol_dct
