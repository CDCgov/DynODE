"""Fit a series of prior parameter distributions through Ordinary Differential Equations (ODEs).

Compare the likelihood of the output to some observed metrics.
"""

import datetime
import json
from typing import Union

import jax.numpy as jnp
import jax.typing
import numpy as np
import numpyro  # type: ignore
from diffrax import Solution  # type: ignore
from jax.random import PRNGKey
from numpyro import distributions as Dist
from numpyro.infer import MCMC, NUTS  # type: ignore
from .abstract_parameters import AbstractParameters
from .config import Config
from .mechanistic_runner import MechanisticRunner
from .typing import SEIC_Compartments
from .utils import date_to_sim_day


class MechanisticInferer(AbstractParameters):
    """Manage the fitting process of epidemiological parameters on ODEs.

    Taking in priors, sampling from their distributions,
    managing MCMC or the sampling/fitting proceedure of choice,
    and coordinating the parsing and use of the posterier distributions.
    """

    def __init__(
        self,
        global_variables_path: str,
        distributions_path: str,
        runner: MechanisticRunner,
        initial_state: SEIC_Compartments,
    ):
        """Initialize an inferer object with config JSONS, a set of ODEs, and an initial state.

        Parameters
        ----------
        global_variables_path : str
            Path to global JSON for parameters shared across all components
            of the model.
        distributions_path : str
            Path to inferer specific JSON of parameters containing prior distributions.
        runner : MechanisticRunner
            Runner class to solve ODEs and return infection timeseries.
        initial_state : SEIC_Compartments
            Initial compartment state at t=0.
        """
        distributions_json = open(distributions_path, "r").read()
        global_json = open(global_variables_path, "r").read()
        self.config = Config(global_json).add_file(distributions_json)
        self.runner = runner
        self.INITIAL_STATE = initial_state
        self.infer_complete = False
        # set inference algo to mcmc
        self.inference_algo = self.set_infer_algo()
        # retrieve population age distribution via passed initial state
        self.config.POPULATION = self.retrieve_population_counts()
        # load all vaccination splines
        (
            self.config.VACCINATION_MODEL_KNOTS,
            self.config.VACCINATION_MODEL_KNOT_LOCATIONS,
            self.config.VACCINATION_MODEL_BASE_EQUATIONS,
        ) = self.load_vaccination_model()
        self.config.CONTACT_MATRIX = self.load_contact_matrix()

    def set_infer_algo(self, inferer_type: str = "mcmc") -> MCMC:
        """Set inference algorithm with attached sampler.

        Parameters
        ----------
        inferer_type : str, optional
            infer algo you wish to use, by default "mcmc"

        Returns
        -------
        MCMC
            returns MCMC inference algorithm as it is the only supported
            algorithm currently

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
            return MCMC(
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
        """Generate post-hoc predictions from solved timeseries in `Solution`.

        Optionally use parameters if sampling variables used for generating predictions.

        Parameters
        ----------
        parameters : dict
            Parameters object returned by `self.get_parameters()` if needed
            to produce predictions for likelihood.
        solution : Solution
            Solution object returned by `self._solve_runner()` or any
            call to `self.runner.run()` containing compartment timeseries

        Returns
        -------
        jax.Array or tuple[jax.Array]
            one or more jax arrays representing the different
            post-hoc predictions generated from `solution`. In this case
            only hospitalizations are returned.
        """
        # add 1 to idxs because we are stratified by time in the solution object
        # sum down to just time x age bins
        assert (
            solution.ys is not None
        ), "solution.ys returned None, odes failed."
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
    ) -> dict[str, Solution | jax.Array | dict]:
        """Solves ODEs and package result together with post-hoc predictions.

        Parameters
        ----------
        tf : int
            number of days to run simulation for

        Returns
        -------
        dict[str, Solution | jax.Array | dict]
            dictionary containing following key value pairs:

            solution: `diffrax.Solution` object of simulation timeseries
            hospitalizations : `jax.Array` return value from
            `self._get_predictions()`
            parameters : `dict` result of `self.get_parameters()` passed to the
            runner to generate the `diffrax.Solution` object.
        """
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
        """Sample likelihood of observed metrics given suite of parameter values.

        Given some observed metrics, samples the likelihood of them occuring
        under a set of parameter distributions sampled by self.inference_algo.
        Currently expects hospitalization data and samples IHR.

        Parameters
        ----------
        tf : int
            days to run simulation for before comparing to obs_metrics
        obs_metrics : jax.Array
            observed data, currently expecting hospitalization data

        Returns
        -------
        None
        """
        dct = self.run_simulation(tf)
        solution = dct["solution"]
        predicted_metrics = dct["hospitalizations"]
        assert isinstance(predicted_metrics, jax.Array)
        assert isinstance(solution, Solution)
        self._checkpoint_compartment_sizes(solution)
        predicted_metrics = jnp.maximum(predicted_metrics, 1e-6)
        numpyro.sample(
            "incidence",
            Dist.Poisson(predicted_metrics),
            obs=obs_metrics,
        )

    def infer(self, obs_metrics: jax.Array) -> MCMC:
        """Infer parameters given priors inside of self.config.

        Parameters
        ----------
        obs_metrics: jax.Array
            observed metrics on which likelihood will be calculated on
            to tune parameters.

        Returns
        -------
        MCMC
            The inference object, currently `numpyro.infer.MCMC`,
            used to infer parameters and produce posterior samples.
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

    def _checkpoint_compartment_sizes(self, solution: Solution):
        """Take note of compartment sizes at end of `solution` and on key dates.

        Saves requested dates from `self.config.COMPARTMENT_SAVE_DATES` if the
        parameter exists. Skipping over any invalid dates.

        This method does not actually save the compartment sizes to a file,
        instead it stores the values within `self.inference_algo.get_samples()`
        so that they may be later saved by self.checkpoint() or by the user.

        Parameters
        ----------
        solution : diffrax.Solution
            a diffrax Solution object returned by solving ODEs, most often
            retrieved by `self.run_simulation()`
        """
        assert (
            solution.ys is not None
        ), "solution.ys returned None, odes failed."
        for compartment in self.config.COMPARTMENT_IDX:
            numpyro.deterministic(
                "final_timestep_%s" % compartment.name,
                solution.ys[compartment][-1],
            )
        for d in getattr(self.config, "COMPARTMENT_SAVE_DATES", []):
            date: datetime.date = d
            date_str = date.strftime("%Y_%m_%d")
            sim_day = date_to_sim_day(date, self.config.INIT_DATE)
            # ensure user requests a day we actually have in `solution`
            if sim_day >= 0 and sim_day < len(
                solution.ys[self.config.COMPARTMENT_IDX.S]
            ):
                for compartment in self.config.COMPARTMENT_IDX:
                    numpyro.deterministic(
                        "%s_timestep_%s" % (date_str, compartment.name),
                        solution.ys[compartment][sim_day],
                    )

    def checkpoint(
        self, checkpoint_path: str, group_by_chain: bool = True
    ) -> None:
        """Save the posterior samples from `self.inference_algo`.

        Saves samples into `checkpoint_path` as a json file.
        will save anything sampled or numpyro.deterministic as
        long as it is within the numpyro trace.

        Parameters
        ----------
        checkpoint_path: str
            a path to which the json file is saved to. Throws error if folder
            does not exist, overwrites existing JSON files within.

        group_by_chain: bool, Optional
            whether or not saved JSON should retain chain/sample structure
            or flatten all chains together into a single list of samples.
            Default, True which retains chain structure creating 2d lists.

        Raises
        ------
        ValueError
            if inference has not been called (not self.infer_complete),
            and thus there are no posteriors to be saved to `checkpoint_path`

        Returns
        -------
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
        """Simulate posterior particles without full inference flow.

        Particles are identified by a (chain, particle) indexing tuple.

        if `external_particle` is specified uses that dict instead of
        self.inference_algo.get_samples() to load numpyro sites.

        Parameters
        ----------
        particles: Union[tuple[int, int], list[tuple[int, int]]]
            a single tuple or list of tuples, each of which specifies
            the (chain_num, particle_num) to load
            will error if values are out of range of what was sampled.
        tf: Union[int, None]:
            number of days to run posterior model for,
            defaults to same number of days used in fitting, if possible.
        external_particle: dict
            for use of particles defined somewhere outside of this class
            instance. For example, loading a checkpoint.json containing saved
            posteriors from a different run. Expects keys that match sampled
            sites within `get_parameters()`
        verbose: bool, optional
            whether or not to pring out the current
            (chain, particle) value being loaded.


        Returns
        -------
        `dict[tuple(int, int)]` a dictionary containing
        the returned value of `self.run_simulation()` evaluated with values
        from (chain_num, particle_num).
        Posterior values used appended to the dictionary under the "posteriors" key.

        Examples
        --------
        <insert 2 chain inference above>
        `load_posterior_particle([(0, 100), [1, 120],...]) = {(0, 100): {solution: diffrax.Solution, "posteriors": {...}},
                                                     (1, 120): {solution: diffrax.Solution, "posteriors": {...}} ...}`

        Notes
        -----
        In the scenario this instance of `MechanisticInferer.run_simulation()`
        samples parameters not named in `external_particle`
        they will be resampled according to the prior specified in self.config.

        Some RNG salt is applied to each (chain, particle) pair so samples
        missing from `external_particle` are sampled different for each particle.

        This may be useful to you if you wish to fit upon some data, then
        vary a new parameter over the posteriors (often during projection).
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
        """Execute `self.run_simulation()` on a single posterior particle.

        Used by `load_posterior_particle`.
        Dont touch unless you know what you are doing.

        Parameters
        ----------
        single_particle : dict[str, jax.typing.ArrayLike]
            a dictionary linking a parameter name to its posterior value,
            a single value or list depending on the sampled parameter
        tf : int
            the number of days to run the simulation for
        chain_paricle_seed : int
            some salting unique to the particle being run,
            used to randomize any NEW parameters sampled that are
            not within `single_particle`

        Returns
        -------
        dict[str, jax.Array| Solution | dict]
            a solution_dict containing the return value of
            `self.run_simulation()` as well as a key "posteriors".
            "posteriors" contains the values within `single_particle` as well
            as any newly sampled values created by `self.run_simulation()`
            that were not found in `single_particle`
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
