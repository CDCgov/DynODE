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
        previous_inferer: MCMC = None,
    ):
        distributions_json = open(distributions_path, "r").read()
        global_json = open(global_variables_path, "r").read()
        self.config = Config(global_json).add_file(distributions_json)
        self.runner = runner
        self.INITIAL_STATE = initial_state
        self.set_infer_algo(previous_inferer=previous_inferer)
        self.retrieve_population_counts()
        self.load_cross_immunity_matrix()
        self.load_vaccination_model()
        self.load_contact_matrix()

    def set_infer_algo(self, previous_inferer=None, inferer_type="mcmc"):
        """
        Sets the inferer's inference algorithm and sampler.
        If passed a previous inferer of the same inferer_type, uses posteriors to aid in the definition of new priors.
        This does require special configuration parameters to aid in transition between sequential inferers.

        Parameters
        ----------
        previous_inferer: None, numpyro.infer.MCMC
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
            if previous_inferer is not None:
                # may want to look into this here:
                # https://num.pyro.ai/en/stable/mcmc.html#id7
                assert isinstance(
                    previous_inferer, MCMC
                ), "the previous inferer is not of the same type."
                self.inference_algo.post_warmup_state = (
                    previous_inferer.last_state
                )

    def likelihood(self, obs_metrics):
        """
        Given some observed metrics, samples the likelihood of them occuring
        under a set of parameter distributions sampled by self.inference_algo.

        Currently expects hospitalization data and samples IHR using a negative binomial distribution.

        Parameters
        -----------
        obs_metrics: jnp.ndarray
            the observed metrics on which likelihood is calculated. Usually synthetic or empirical data.

        Returns
        -----------
        None
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

        # scale model_incidence by the ihr, then apply NB observation model
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
        """
        for key, param in parameters.items():
            if issubclass(type(param), Dist.Distribution):
                param = numpyro.sample(key, param)
            elif isinstance(param, (np.ndarray, list)):
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
            parameters[key] = param
        return parameters

    def get_parameters(self):
        """
        Goes through the parameters passed to the inferer, if they are distributions, it samples them.
        Otherwise it returns their raw values.

        Returns a dictionary of {str:obj} where obj may either be a float value,
        or a jax tracer (in the case of a sampled value). Finally converts all list types to jax tracers for inference.
        """
        parameters = {
            "CONTACT_MATRIX": self.config.CONTACT_MATRIX,
            "POPULATION": self.config.POPULATION,
            "NUM_STRAINS": self.config.NUM_STRAINS,
            "NUM_AGE_GROUPS": self.config.NUM_AGE_GROUPS,
            "NUM_WANING_COMPARTMENTS": self.config.NUM_WANING_COMPARTMENTS,
            "WANING_PROTECTIONS": self.config.WANING_PROTECTIONS,
            "MAX_VAX_COUNT": self.config.MAX_VAX_COUNT,
            "CROSSIMMUNITY_MATRIX": self.config.CROSSIMMUNITY_MATRIX,
            "VAX_EFF_MATRIX": self.config.VAX_EFF_MATRIX,
            "BETA_TIMES": self.config.BETA_TIMES,
            "STRAIN_R0s": self.config.STRAIN_R0s,
            "INFECTIOUS_PERIOD": self.config.INFECTIOUS_PERIOD,
            "EXPOSED_TO_INFECTIOUS": self.config.EXPOSED_TO_INFECTIOUS,
            "INTRODUCTION_TIMES": self.config.INTRODUCTION_TIMES,
        }
        parameters = self.sample_if_distribution(parameters)
        # if we are sampling external introductions, we must reload the function
        self.load_external_i_distributions(parameters["INTRODUCTION_TIMES"])
        beta = parameters["STRAIN_R0s"] / parameters["INFECTIOUS_PERIOD"]
        gamma = 1 / parameters["INFECTIOUS_PERIOD"]
        sigma = 1 / parameters["EXPOSED_TO_INFECTIOUS"]
        # since our last waning time is zero to account for last compartment never waning
        # we include an if else statement to catch a division by zero error here.
        waning_rates = np.array(
            [
                1 / waning_time if waning_time > 0 else 0
                for waning_time in self.config.WANING_TIMES
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
        # model only expects jax lists, so replace all lists and numpy arrays with lists here.
        for key, val in parameters.items():
            if isinstance(val, (np.ndarray, list)):
                parameters[key] = jnp.array(val)

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
