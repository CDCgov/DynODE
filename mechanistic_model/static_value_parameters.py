from mechanistic_model.abstract_parameters import AbstractParameters
from config.config import Config
import numpyro
import numpyro.distributions as Dist
from functools import partial


class StaticValueParameters(AbstractParameters):

    def __init__(
        self, INITIAL_STATE, runner_config_path, global_variables_path
    ):
        self.__dict__ = (
            Config(global_variables_path).add_file(runner_config_path).__dict__
        )
        self.INITIAL_STATE = INITIAL_STATE
        self.retrieve_population_counts()
        self.load_cross_immunity_matrix()
        self.load_vaccination_model()
        self.load_external_i_distributions()
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
        args = {
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
        }
        beta = self.STRAIN_R0s / self.INFECTIOUS_PERIOD
        gamma = 1 / self.INFECTIOUS_PERIOD
        sigma = 1 / self.EXPOSED_TO_INFECTIOUS
        # since our last waning time is zero to account for last compartment never waning
        # we include an if else statement to catch a division by zero error here.
        waning_rates = [
            1 / waning_time if waning_time > 0 else 0
            for waning_time in self.WANING_TIMES
        ]
        # add final parameters, if your model expects added parameters, add them here
        args = dict(
            args,
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
        return args
