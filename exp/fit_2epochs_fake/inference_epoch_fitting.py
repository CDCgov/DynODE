# %%
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as Dist
import pandas as pd

from mechanistic_model.covid_initializer import CovidInitializer
from mechanistic_model.mechanistic_inferer import MechanisticInferer
from mechanistic_model.mechanistic_runner import MechanisticRunner
from mechanistic_model.static_value_parameters import StaticValueParameters
from model_odes.seip_model import seip_ode

# %%

EXP_ROOT_PATH = "exp/fit_2epochs_fake/"
GLOBAL_EPOCH1_CONFIG_PATH = EXP_ROOT_PATH + "config_global_epoch1.json"
GLOBAL_EPOCH2_CONFIG_PATH = EXP_ROOT_PATH + "config_global_epoch2.json"
INITIALIZER_CONFIG_PATH = EXP_ROOT_PATH + "config_initializer.json"
RUNNER_CONFIG_PATH = EXP_ROOT_PATH + "config_runner_synthetic.json"
INFERER_EPOCH1_PATH = EXP_ROOT_PATH + "config_inferer_covid_epoch_1.json"
INFERER_EPOCH2_PATH = EXP_ROOT_PATH + "config_inferer_covid_epoch_2.json"
SYNTHETIC_DATA_PATH = EXP_ROOT_PATH + "synthetic_dat.csv"
# %%
synthetic_data = pd.read_csv(SYNTHETIC_DATA_PATH, index_col="time")
epoch_1_synthetic_hosp_data = synthetic_data[0:241]
epoch_2_synthetic_hosp_data = synthetic_data[240:]
print(epoch_1_synthetic_hosp_data.head())

# %%
EXPLANATION_OF_SYNTHETIC_DATA = (
    """A COUPLE OF NOTES ABOUT THE SYNTHETIC DATA WE ARE USING.
      We are running with 4 synthetic strains, pre-omi, omi, ba2/5, XBB
Their True R0s are fixed at 1.2, 2.4, 2.7, and 2.5 respectively.
To see all of the parameters used to generate the synthetic data, see %s.

This file will attempt to fit inferers on this data, with the first
fitting upon the first 240 days, and the second fitting on the latter 210 days.
Both of these inference epochs will introduce one strain, in epoch 1 BA2/5 hits
peak introduction rate on day 70, while in epoch 2 it happens on day 30 (actual day 270)
Be mindful of what t=0 means for each inferer, as it depends on their start date.

Another note: The runner specifies strain interactions and vax_eff matrix,
when combining strains in epoch 2, the more recent of the 2 strain's cell is used,
no averaging of the two strains occurs
"""
    % RUNNER_CONFIG_PATH
)
print(EXPLANATION_OF_SYNTHETIC_DATA)
# %%
# begins at t=0
initializer = CovidInitializer(
    INITIALIZER_CONFIG_PATH, GLOBAL_EPOCH1_CONFIG_PATH
)

runner = MechanisticRunner(seip_ode)

inferer = MechanisticInferer(
    GLOBAL_EPOCH1_CONFIG_PATH,
    INFERER_EPOCH1_PATH,
    runner,
    initializer.get_initial_state(),
)
# %%
mc1 = inferer.infer(epoch_1_synthetic_hosp_data.to_numpy())


# %%
samp = mc1.get_samples(group_by_chain=True)
fig, axs = plt.subplots(2, 2)
axs[0, 0].set_title("Intro Time")
axs[0, 0].plot(np.transpose(samp["INTRODUCTION_TIMES_0"]))
axs[0, 1].set_title("BA2/BA5 R0")
axs[0, 1].plot(np.transpose(samp["STRAIN_R0s_2"]))
axs[1, 0].set_title("ihr-50-64")
axs[1, 0].plot(np.transpose(samp["ihr"][:, :, 2]))
axs[1, 1].set_title("ihr-65+")
axs[1, 1].plot(np.transpose(samp["ihr"][:, :, 3]), label=range(0, 4))
# axs[2, 0].set_title("k")
# axs[2, 0].plot(np.transpose(samp["k"]))
fig.legend()
plt.show()
print(
    "lets look for any clearly diverging chains, see if we need to drop anything"
)
print(
    """Something super interesting here, the fake data is generated with Poisson distribution,
      while we sample incidence via negative binomial. As K gets larger NB approaches poisson,
      this is why chain 2 is divergent but actually more correct!"""
)


# %%
chain_2_intro_time = samp["INTRODUCTION_TIMES_0"][2, :]
chain_2_ba2_r0 = samp["STRAIN_R0s_2"][2, :]
print(
    "PARAMETERS FIXED FOR GENERATING EPOCH 2 INITIAL STATE: "
    "INTRO_TIME = [%d]   BA2/5 R0: %s"
    % (np.median(chain_2_intro_time), str(np.median(chain_2_ba2_r0)))
)
# %%
# for now we will take the median values fitted and use them to create a final state
# see config_runner_epoch1_end.json to see the medians
MEDIAN_EPOCH_1_STATIC_PARAMS_PATH = (
    EXP_ROOT_PATH + "config_runner_epoch1_end.json"
)
median_epoch_1_fitted_params = StaticValueParameters(
    initializer.get_initial_state(),
    MEDIAN_EPOCH_1_STATIC_PARAMS_PATH,
    GLOBAL_EPOCH1_CONFIG_PATH,
)
s_shape = median_epoch_1_fitted_params.INITIAL_STATE[0].shape
rest_shape = median_epoch_1_fitted_params.INITIAL_STATE[1].shape
all_final_states = [
    [],
    [],
    [],
    [],
]
# total_pops = [0, 0, 0, 0]
for intro_time, ba2_r0 in zip(chain_2_intro_time, chain_2_ba2_r0):
    median_epoch_1_fitted_params.config.STRAIN_R0s[2] = ba2_r0
    median_epoch_1_fitted_params.config.INTRODUCTION_TIMES[0] = intro_time
    median_epoch_1_fitted_params.load_external_i_distributions(
        median_epoch_1_fitted_params.config.INTRODUCTION_TIMES
    )
    run_with_fitted_params_epoch_1 = runner.run(
        median_epoch_1_fitted_params.INITIAL_STATE,
        median_epoch_1_fitted_params.get_parameters(),
    )
    for i, compartment in enumerate(run_with_fitted_params_epoch_1.ys):
        # timestep is first dimension, so we selecting compartment at t=-1
        # which in this case is t=240
        last_time_step = compartment[-1]
        # add a rolling sum to the average_final_state variable
        all_final_states[i].append(last_time_step)
        # total_pops[i] += np.sum(last_time_step)
# %%
average_final_state = [
    np.mean(compartment, axis=0) for compartment in all_final_states
]
print("Total Pop of the new average final state")
print(
    np.sum([np.sum(compartment) for compartment in average_final_state[:-1]])
)
print("Total Pop of the initial state of the sim, should be unchanged")
print(
    np.sum(
        [
            np.sum(compartment)
            for compartment in initializer.get_initial_state()
        ]
    )
)

# %%


class PosteriorInferer(MechanisticInferer):
    def __init__(
        self,
        global_variables_path: str,
        distributions_path: str,
        runner: MechanisticRunner,
        initial_state: tuple,
        prior_inferer=None,
    ):
        super().__init__(
            global_variables_path,
            distributions_path,
            runner,
            initial_state,
            prior_inferer,
        )

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
        # we are not using posteriors for the BA2 R0 or the intro time
        with numpyro.plate(
            "num_parameters", len(self.prior_inferer_param_names)
        ):
            u = numpyro.sample("us", Dist.Normal(0, 1))
        multi_samples = [
            numpyro.deterministic(
                self.prior_inferer_param_names[i],
                self.prior_inferer_particle_means[i]
                + sum(
                    self.cholesky_triangle_matrix[i, j] * u[j]
                    for j in range(i + 1)
                ),
            )
            for i in range(len(self.prior_inferer_param_names))
        ]
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
                "ihr": multi_samples[:4],
            }
        )
        # model only expects jax lists, so replace all lists and numpy arrays with lists here.
        for key, val in parameters.items():
            if isinstance(val, (np.ndarray, list)):
                parameters[key] = jnp.array(val)

        return parameters

    def likelihood(self, obs_metrics):
        parameters = self.get_parameters()
        solution = self.runner.run(
            self.INITIAL_STATE, args=parameters, tf=len(obs_metrics)
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
        numpyro.sample(
            "incidence",
            Dist.Poisson(model_incidence * parameters["ihr"]),
            obs=obs_metrics,
        )


# %%
# Basically the same but with some minor changes to how sampling is done
inferer2 = PosteriorInferer(
    GLOBAL_EPOCH2_CONFIG_PATH,
    INFERER_EPOCH2_PATH,
    runner,
    tuple(average_final_state),
    prior_inferer=mc1,
)
print(inferer2.cholesky_triangle_matrix)
print(inferer2.prior_inferer_particle_means)

# %%
print(np.sum(inferer.vaccination_rate(0)))
print(np.sum(inferer2.vaccination_rate(0)))
problems = """
                             Problems Faced
1) the global config specified 4 strains instead of 3,
should be fine because this was needed to create the synthetic data

2) have to re-add the delay on the vax model when infer multiple epochs with the same vax function

"""
# %%
print(inferer2.prior_inferer_param_names)

# %%
mc2 = inferer2.infer(epoch_2_synthetic_hosp_data.to_numpy())

# %%
mc2.print_summary(exclude_deterministic=False)
# %%
