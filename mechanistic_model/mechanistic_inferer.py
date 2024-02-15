from functools import partial
from mechanistic_model.mechanistic_runner import MechanisticRunner
from mechanistic_model.abstract_parameters import AbstractParameters
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import pandas as pd
from diffrax import (  # Solution,
    ODETerm,
    PIDController,
    SaveAt,
    Tsit5,
    diffeqsolve,
)
from jax.random import PRNGKey
from numpyro import distributions as Dist
from numpyro.infer import MCMC, NUTS

import utils
from config.config import Config


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
        self.__dict__ = (
            Config(global_variables_path).add_file(distributions_path).__dict__
        )
        self.runner = runner
        self.INITIAL_STATE = initial_state
        self.set_infer_algo()
        self.retrieve_population_counts()
        self.load_cross_immunity_matrix()
        self.load_vaccination_model()
        self.load_external_i_distributions()
        self.load_contact_matrix()

    def set_infer_algo(self, inferer_type="MCMC"):
        if inferer_type == "MCMC":
            self.inference_algo = MCMC(
                NUTS(
                    self.likelihood,
                    dense_mass=True,
                    max_tree_depth=5,
                    init_strategy=numpyro.infer.init_to_median,
                ),
                num_warmup=self.INFERENCE_NUM_WARMUP,
                num_samples=self.INFERENCE_NUM_SAMPLES,
                num_chains=self.INFERENCE_NUM_CHAINS,
                progress_bar=self.INFERENCE_PROGRESS_BAR,
            )
        else:
            raise NotImplementedError(
                "Inference algorithms other than MCMC have not been implemented yet, try overriding the method in a subclass to set self.inference_algo"
            )

    def likelihood(self, obs_metrics):
        """
        given some observed metrics, samples the liklihood of them occuring under a set of parameter distributions sampled by self.inference_algo.

        Currently expects Hospitalization data and samples IHR using a negative binomial distribution.
        """
        solution = self.runner.run(
            self.INITIAL_STATE, args=self.get_parameters()
        )
        # add 1 to idxs because we are straified by time in the solution object
        # sum down to just time x age bins
        model_incidence = jnp.sum(
            solution.ys[self.COMPARTMENT_IDX.C],
            axis=(
                self.I_AXIS_IDX.hist + 1,
                self.I_AXIS_IDX.vax + 1,
                self.I_AXIS_IDX.strain + 1,
            ),
        )
        # axis = 0 because we take diff across time
        model_incidence = jnp.diff(model_incidence, axis=0)

        # sample infection hospitalization rate here
        with numpyro.plate("num_age", self.NUM_AGE_GROUPS):
            ihr = numpyro.sample("ihr", Dist.Beta(0.5, 10))

        # scale model_incidence w ihr and apply NB observation model
        k = numpyro.sample("k", Dist.HalfCauchy(1.0))
        numpyro.sample(
            "incidence",
            Dist.NegativeBinomial2(
                mean=model_incidence * ihr, concentration=k
            ),
            obs=obs_metrics,
        )

    def get_parameters(self):
        """
        Goes through the parameters passed to the inferer, if they are distributions, it samples them. Otherwise it returns their raw values.
        Returns a dictionary of {str:obj} where obj may either be a float value, or a jax tracer (in the case of a sampled value).
        All code which uses values from sample_distributions must use JAX operations to work with these values.
        """
        parameters = {
            "CONTACT_MATRIX": self.CONTACT_MATRIX,
            "POPULATION": self.POPULATION,
            "NUM_STRAINS": self.NUM_STRAINS,
            "NUM_AGE_GROUPS": self.NUM_AGE_GROUPS,
            "NUM_WANING_COMPARTMENTS": self.NUM_WANING_COMPARTMENTS,
            "WANING_PROTECTIONS": self.WANING_PROTECTIONS,
            "MAX_VAX_COUNT": self.MAX_VAX_COUNT,
            "CROSSIMMUNITY_MATRIX": self.CROSSIMMUNITY_MATRIX,
            "VAX_EFF_MATRIX": self.VAX_EFF_MATRIX,
            "BETA_TIMES": self.BETA_TIMES,
            "STRAIN_R0s": self.STRAIN_R0s,
            "INFECTIOUS_PERIOD": self.INFECTIOUS_PERIOD,
            "EXPOSED_TO_INFECTIOUS": self.EXPOSED_TO_INFECTIOUS,
        }
        for key, param in parameters.items():
            if issubclass(type(param), Dist.Distribution):
                param = numpyro.sample(key, param)
            elif isinstance(param, np.ndarray):
                param = jnp.array(
                    [
                        (
                            numpyro.sample(key + "_" + str(i), val)
                            if issubclass(type(val), Dist.Distribution)
                            else val
                        )
                        for i, val in enumerate(param)
                    ]
                )
            parameters[key] = param
        beta = parameters["STRAIN_R0s"] / parameters["INFECTIOUS_PERIOD"]
        gamma = 1 / parameters["INFECTIOUS_PERIOD"]
        sigma = 1 / parameters["EXPOSED_TO_INFECTIOUS"]
        # since our last waning time is zero to account for last compartment never waning
        # we include an if else statement to catch a division by zero error here.
        waning_rates = np.array(
            [
                1 / waning_time if waning_time > 0 else 0
                for waning_time in self.WANING_TIMES
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
                "EXTERNAL_I": partial(self.external_i),
                "VACCINATION_RATES": self.vaccination_rate,
                "BETA_COEF": self.beta_coef,
            }
        )
        for key, val in parameters.items():
            if isinstance(val, (np.ndarray, list)):
                parameters[key] = jnp.array(val)

        return parameters

    def infer(self, obs_metrics):
        """
        Infer parameters given priors inside of self.distributions, returns an inference_algo object with posterior distributions for each sampled parameter.
        Parameters
        -----------
        obs_metrics: jnp.array
            observed metrics on which likelihood will be calculated on to tune parameters.

        Returns
        -----------
        an inference object, often numpyro.infer.MCMC object used to infer parameters.
        This can be used to print summaries, pass along covariance matrices, or query posterier distributions
        """
        infer_obj = self.inference_algo.run(
            rng_key=PRNGKey(self.INFERENCE_PRNGKEY),
            obs_metrics=obs_metrics,
        )
        infer_obj.print_summary()
        return infer_obj
