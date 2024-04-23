import jax.numpy as jnp
import numpy as np

from config.config import Config
from mechanistic_model.abstract_parameters import AbstractParameters


class StaticValueParameters(AbstractParameters):
    def __init__(
        self, INITIAL_STATE, runner_config_path, global_variables_path
    ):
        runner_json = open(runner_config_path, "r").read()
        global_json = open(global_variables_path, "r").read()
        self.config = Config(global_json).add_file(runner_json)
        self.INITIAL_STATE = INITIAL_STATE
        self.retrieve_population_counts()
        self.load_cross_immunity_matrix()
        self.load_vaccination_model()
        self.load_contact_matrix()

    def get_parameters(
        self,
    ):
        """
        A function that returns model args as a dictionary as expected by the ODETerm function f(t, y(t), args)dt
        https://docs.kidger.site/diffrax/api/terms/#diffrax.ODETerm

        for example functions f() in charge of disease dynamics see the model_odes folder.

        Parameters
        ----------
        `sample`: boolean
            whether or not to sample key parameters, used when model is being run in MCMC and parameters are being infered
        `sample_dist_dict`: dict(str:numpyro.distribution)
            a dictionary of parameters to sample.
            follows format "parameter_name":numpyro.Distributions.dist(). DO NOT pass numpyro.sample() objects to the dictionary.

        Returns
        ----------
        dict{str: Object}: A dictionary where key value pairs are used as parameters by an ODE model
        """
        # get counts of the initial state compartments by age bin.
        # ignore the C compartment since it is just house keeping
        # TODO abstract this away for code reuse.
        args = {
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
            "CONSTANT_STEP_SIZE": self.config.CONSTANT_STEP_SIZE,
            "INTRODUCTION_TIMES": self.config.INTRODUCTION_TIMES,
            "INTRODUCTION_SCALES": self.config.INTRODUCTION_SCALES,
            "INTRODUCTION_PERCS": self.config.INTRODUCTION_PERCS,
            "MIN_HOMOLOGOUS_IMMUNITY": self.config.MIN_HOMOLOGOUS_IMMUNITY,
        }
        beta = self.config.STRAIN_R0s / self.config.INFECTIOUS_PERIOD
        gamma = 1 / self.config.INFECTIOUS_PERIOD
        sigma = 1 / self.config.EXPOSED_TO_INFECTIOUS
        # since our last waning time is zero to account for last compartment never waning
        # we include an if else statement to catch a division by zero error here.
        waning_rates = np.array(
            [
                1 / waning_time if waning_time > 0 else 0
                for waning_time in self.config.WANING_TIMES
            ]
        )
        # add final parameters, if your model expects added parameters, add them here
        args = dict(
            args,
            **{
                "BETA": beta,
                "SIGMA": sigma,
                "GAMMA": gamma,
                "WANING_RATES": waning_rates,
                "EXTERNAL_I": self.external_i,
                "VACCINATION_RATES": self.vaccination_rate,
                "BETA_COEF": self.beta_coef,
                "SEASONAL_VACCINATION_RESET": self.seasonal_vaccination_reset,
            }
        )
        for key, val in args.items():
            if isinstance(val, (np.ndarray, list)):
                args[key] = jnp.array(val)

        return args
