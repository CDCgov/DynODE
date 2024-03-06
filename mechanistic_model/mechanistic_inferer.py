import copy

import jax.numpy as jnp
import numpy as np
import numpyro
from jax.random import PRNGKey
from numpyro import distributions as Dist
from numpyro.infer import MCMC, NUTS

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
    ):
        distributions_json = open(distributions_path, "r").read()
        global_json = open(global_variables_path, "r").read()
        self.config = Config(global_json).add_file(distributions_json)
        self.runner = runner
        self.INITIAL_STATE = initial_state
        self.set_infer_algo()
        self.retrieve_population_counts()
        self.load_cross_immunity_matrix()
        self.load_vaccination_model()
        self.load_contact_matrix()

    def set_infer_algo(self, inferer_type="mcmc"):
        """
        Sets the inferer's inference algorithm and sampler.
        """
        supported_infer_algos = ["mcmc"]
        if inferer_type.lower().strip() not in supported_infer_algos:
            raise NotImplementedError(
                "Inference algorithms other than MCMC have not been implemented yet,"
                "try overriding the method in a subclass to set self.inference_algo"
            )

        if inferer_type == "mcmc":
            self.inference_algo = MCMC(
                NUTS(
                    self.likelihood,
                    dense_mass=True,
                    max_tree_depth=5,
                    init_strategy=numpyro.infer.init_to_median,
                ),
                num_warmup=self.config.INFERENCE_NUM_WARMUP,
                num_samples=self.config.INFERENCE_NUM_SAMPLES,
                num_chains=self.config.INFERENCE_NUM_CHAINS,
                progress_bar=self.config.INFERENCE_PROGRESS_BAR,
            )

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

        k = numpyro.sample("k", Dist.HalfCauchy(1.0))
        numpyro.sample(
            "incidence",
            Dist.NegativeBinomial2(
                mean=model_incidence * ihr, concentration=k
            ),
            obs=obs_metrics,
        )

    def sample_if_distribution(self, parameters):
        """
        given a dictionary of keys and parameters, searches through all keys
        and samples the distribution associated with that key, if it exists.
        Otherwise returns the constant value associated with that key.
        Converts lists with distributions inside to `jnp.ndarray`

        Parameters
        ----------
        parameters: dict{str: obj}
            a dictionary mapping a parameter name to an object, either a value or a distribution.
            `numpyro.distribution` objects are sampled, and their sampled value replaces the distribution object
            within parameters. Capable of sampling lists with static values and distributions together.

        Returns
        ----------
        parameters_cpy: a new dictionary with any `numpyro.distribution` objects replaced with jax.tracer samples
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

        Returns a dictionary of {str:obj} where obj may either be a float value,
        or a jax tracer (in the case of a sampled value).
        Converts all list types to jax tracers if values within are sampled.
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
        beta = parameters["STRAIN_R0s"] / parameters["INFECTIOUS_PERIOD"]
        gamma = 1 / parameters["INFECTIOUS_PERIOD"]
        sigma = 1 / parameters["EXPOSED_TO_INFECTIOUS"]
        # since our last waning time is zero to account for last compartment never waning
        # we include an if else statement to catch a division by zero error here.
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
