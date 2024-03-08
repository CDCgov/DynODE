# %%
import copy

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as Dist
import pandas as pd
from tqdm import tqdm

import utils
from mechanistic_model.covid_initializer import CovidInitializer
from mechanistic_model.mechanistic_inferer import MechanisticInferer
from mechanistic_model.mechanistic_runner import MechanisticRunner
from mechanistic_model.solution_iterpreter import SolutionInterpreter
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
# with jax.checking_leaks():
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
samp_flatten = mc1.get_samples(group_by_chain=False)
sample_intro_times = samp_flatten["INTRODUCTION_TIMES_0"]
sample_ba2_r0s = samp_flatten["STRAIN_R0s_2"]
print(
    "PARAMETERS FIXED FOR GENERATING EPOCH 2 INITIAL STATE: "
    "INTRO_TIME = [%s]   BA2/5 R0: %s"
    % (str(np.median(sample_intro_times)), str(np.median(sample_ba2_r0s)))
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
median_epoch1_solution = runner.run(
    median_epoch_1_fitted_params.INITIAL_STATE,
    median_epoch_1_fitted_params.get_parameters(),
    tf=len(epoch_1_synthetic_hosp_data),
)

interpreter = SolutionInterpreter(
    median_epoch1_solution,
    GLOBAL_EPOCH1_CONFIG_PATH,
    GLOBAL_EPOCH1_CONFIG_PATH,
)
interpreter.summarize_solution()
# %%
# now we get an average final state by running all sampled parameters
# as static parameters and averaging the final state of each.
s_shape = median_epoch_1_fitted_params.INITIAL_STATE[0].shape
rest_shape = median_epoch_1_fitted_params.INITIAL_STATE[1].shape
all_final_states = [
    [],
    [],
    [],
    [],
]
# total_pops = [0, 0, 0, 0]
for intro_time, ba2_r0 in tqdm(
    zip(sample_intro_times, sample_ba2_r0s), total=len(sample_intro_times)
):
    median_epoch_1_fitted_params.config.STRAIN_R0s[2] = ba2_r0
    median_epoch_1_fitted_params.config.INTRODUCTION_TIMES[0] = intro_time
    run_with_fitted_params_epoch_1 = runner.run(
        median_epoch_1_fitted_params.INITIAL_STATE,
        median_epoch_1_fitted_params.get_parameters(),
        tf=len(epoch_1_synthetic_hosp_data),
    )
    for i, compartment in enumerate(run_with_fitted_params_epoch_1.ys):
        # timestep is first dimension, so we selecting compartment at t=-1
        # which in this case is t=240
        last_time_step = compartment[-1]
        # add a rolling sum to the average_final_state variable
        all_final_states[i].append(last_time_step)
        # total_pops[i] += np.sum(last_time_step)
average_final_state = [
    np.mean(compartment, axis=0) for compartment in all_final_states
]
# %%
# now we need to move people around by collapsing strains for the next epoch
average_final_state = tuple(average_final_state)
average_final_state_combined = []
for idx, compartment in enumerate(average_final_state):
    # if susceptible compartment, no strain axis,set strain_axis=False
    strain_axis = idx != 0
    state_mapping, strain_mapping = utils.combined_strains_mapping(1, 0, 3)
    compartment_collapsed = utils.combine_strains(
        compartment,
        state_mapping=state_mapping,
        strain_mapping=strain_mapping,
        num_strains=3,
        state_dim=1,  # start from 0, 2nd dim = 1
        strain_dim=3,  # start from 0, 4th dim = 3
        strain_axis=strain_axis,
    )
    average_final_state_combined.append(compartment_collapsed)

# %% testing to ensure population did not actually change, people just moved around
print("Total Pop of the new average final state")
print(
    np.sum(
        [
            np.sum(compartment)
            for compartment in average_final_state_combined[:-1]
        ]
    )
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


# %% creating dummy class to do posterior inference of ihrs
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
        jax.debug.print("Strain R0 {x}", x=parameters["STRAIN_R0s"][2])
        jax.debug.print("ihrs {x}", x=multi_samples)
        jax.debug.print("Intro Time: {x}", x=parameters["INTRODUCTION_TIMES"])
        # if we are sampling external introductions, we must reload the function
        # self.load_external_i_distributions(parameters["INTRODUCTION_TIMES"])
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


inferer2 = PosteriorInferer(
    GLOBAL_EPOCH2_CONFIG_PATH,
    INFERER_EPOCH2_PATH,
    runner,
    tuple(average_final_state_combined),
    prior_inferer=mc1,
)
print(inferer2.cholesky_triangle_matrix)
print(inferer2.prior_inferer_particle_means)

# %% run epoch 2 inference to fit on xbb
mc2 = inferer2.infer(epoch_2_synthetic_hosp_data.to_numpy())
mc2.print_summary(exclude_deterministic=False)

# %% plot the fitting by chain for the second epoch
samp2 = mc2.get_samples(group_by_chain=True)
fig, axs = plt.subplots(2, 2)
axs[0, 0].set_title("Intro Time XBB")
axs[0, 0].plot(np.transpose(samp2["INTRODUCTION_TIMES_0"]))
axs[0, 1].set_title("XBB R0")
axs[0, 1].plot(np.transpose(samp2["STRAIN_R0s_2"]))
axs[1, 0].set_title(" ihr-50-64")
axs[1, 0].plot(np.transpose(mc2._states["z"]["ihr_2"]))
axs[1, 1].set_title("ihr-65+")
axs[1, 1].plot(np.transpose(mc2._states["z"]["ihr_3"]), label=range(0, 4))
# axs[2, 0].set_title("k")
# axs[2, 0].plot(np.transpose(samp["k"]))
fig.legend()
plt.show()
# %% run using epoch 2 median parameters
MEDIAN_EPOCH_2_STATIC_PARAMS_PATH = (
    EXP_ROOT_PATH + "config_runner_epoch2_end.json"
)
median_epoch_2_fitted_params = StaticValueParameters(
    tuple(average_final_state_combined),
    MEDIAN_EPOCH_2_STATIC_PARAMS_PATH,
    GLOBAL_EPOCH2_CONFIG_PATH,
)
print(median_epoch_2_fitted_params.config.STRAIN_R0s)
median_epoch2_solution = runner.run(
    median_epoch_2_fitted_params.INITIAL_STATE,
    median_epoch_2_fitted_params.get_parameters(),
    tf=len(epoch_2_synthetic_hosp_data),
)
# %% combine the median runs of both epochs for a final timeline

combined_epochs = utils.combine_epochs(
    [median_epoch1_solution.ys, median_epoch2_solution.ys],
    from_strains=["omicron"],
    to_strains=["delta"],
    strain_idxs=[
        median_epoch_1_fitted_params.config.STRAIN_IDX,
        median_epoch_2_fitted_params.config.STRAIN_IDX,
    ],
    num_tracked_strains=3,
)


# %%
class spoof_solution_class:
    def __init__(self, solys):
        self.ys = solys


interpreter.solution = spoof_solution_class(combined_epochs)
interpreter.STRAIN_IDX = median_epoch_2_fitted_params.config.STRAIN_IDX
fig, ax = interpreter.summarize_solution()
# %%
ihr_epoch_1_median = np.median(
    mc1.get_samples(group_by_chain=False)["ihr"], axis=0
)
ihr_epoch_2_median = np.median(
    np.array([mc2._states["z"]["ihr_" + str(i)] for i in range(4)]),
    axis=(1, 2),
)
print(ihr_epoch_1_median)
print(ihr_epoch_2_median)
for i, (solution, ihrs, hosp) in enumerate(
    zip(
        [median_epoch1_solution, median_epoch2_solution],
        [ihr_epoch_1_median, ihr_epoch_2_median],
        [epoch_1_synthetic_hosp_data, epoch_2_synthetic_hosp_data],
    )
):
    model_incidence = jnp.sum(solution.ys[3], axis=(2, 3, 4))
    model_incidence = jnp.diff(model_incidence, axis=0)
    model_incidence = model_incidence * ihrs
    print(model_incidence.shape)
    print(hosp.shape)
    fig, ax = plt.subplots(1)
    ax.plot(
        list(range(len(model_incidence))),
        model_incidence,
        label=["0-17-infer", "18-49-infer", "50-64-infer", "65+-infer"],
    )
    ax.plot(
        list(range(len(model_incidence))),
        hosp,
        label=["0-17-truth", "18-49-truth", "50-64-truth", "65+-truth"],
    )
    fig.legend()
    # ax.set_title("model vs obs data epoch 1")
    ax.set_title("model vs obs data epoch " + str(i))
    plt.show()
# %%
print(np.sum(inferer2.external_i(0, [30])))
print(np.sum(inferer2.external_i(2, [30])))
print(np.sum(inferer2.external_i(4, [30])))
print(np.sum(inferer2.external_i(30, [30])))

# %%
problems = """
                             Problems Faced
1) the global config specified 4 strains instead of 3,
should be fine because this was needed to create the synthetic data

2) have to re-add the delay on the vax model when infer multiple epochs with the same vax function

3) have to remember that the average compartment must be generated by running the runner for the same number of days as was inferred.

4) dont forget to actually transition the strains when you calculate the average final state, gotta zero out the last index again.
"""
