# %%
import argparse
import copy
import datetime
import json
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as Dist
from matplotlib.backends.backend_pdf import PdfPages

sys.path.append("/app")
sys.path.append("/app/mechanistic_model/")

import utils
from exp.fifty_state_2304_2404_3strain.epoch_two_initializer import \
    smh_initializer_epoch_two
from exp.fifty_state_2304_2404_3strain.inferer_smh import SMHInferer
from mechanistic_model.mechanistic_runner import MechanisticRunner
from model_odes.seip_model import Parameters, seip_ode

"""
 
Scenario A| No booster, low immune escape  | noBoo_lowIE    | A-2024-03-01
Scenario B| No booster, high immune escape | noBoo_highIE   | B-2024-03-01
Scenario C| 65+ booster, low immune escape | 65Boo_lowIE    | C-2024-03-01
Scenario D| 65+ booster, high immune escape| 65Boo_highIE   | D-2024-03-01
Scenario E| All booster, low immune escape | allBoo_lowIE   | E-2024-03-01
Scenario F| All booster, high immune escape| allBoo_highIE  | F-2024-03-01
"""
PROJECTION_DAYS = 365  # projection ends 2025, 4, 27
EPOCH_2_DAYS = 392  # epoch 2 ends 2024, 4, 27
suffix = "v1"
PREV_EPOCH_PATH = "/output/fifty_state_2304_2404_3strain/smh_epoch_2_240514_2"
pdf_filename = f"projections{suffix}.pdf"
pdf_pages = PdfPages(pdf_filename)
EXP_PATH = "/app/exp/projections"
SCENARIOS_LIST = [
    "noBoo_lowIE",
    "noBoo_highIE",
    "65Boo_lowIE",
    "65Boo_highIE",
    "allBoo_lowIE",
    "allBoo_highIE",
]
SCENARIOS_DICT = {
    k: ltr for k, ltr in zip(SCENARIOS_LIST, ["A", "B", "C", "D", "E", "F"])
}
# SCENARIOS_LIST = ["allBoo_lowIE", "allBoo_highIE"]
NUM_PARTICLES_PER_CHAIN = 25
PROJECTION_START_DATE = datetime.datetime(2024, 4, 28)


# %%


def get_new_seed(key):
    _, new_key = jax.random.split(key)
    return new_key


def retrieve_inferer_obs(state):
    # state_config_path = os.path.join(az_output_path, state)
    print("Retrieving " + state + "\n")
    inferer_filename = (
        "config_inferer_used.json"  # "posteriors_untouched.json"
    )
    global_filename = (
        "config_global_used.json"  # "config_global_untouched.json"
    )
    initializer_filename = (
        "config_initializer_used.json"  # config_initializer_sero_template.json
    )
    # use EXP_PATH if you want to run locally
    INFERER_CONFIG_PATH = os.path.join(
        PREV_EPOCH_PATH, "%s/%s" % (state, inferer_filename)
    )
    GLOBAL_CONFIG_PATH = os.path.join(
        PREV_EPOCH_PATH, "%s/%s" % (state, global_filename)
    )
    INITIALIZER_CONFIG_PATH = os.path.join(
        PREV_EPOCH_PATH, "%s/%s" % (state, initializer_filename)
    )
    # sets up the initial conditions, initializer.get_initial_state() passed to runner
    initializer = smh_initializer_epoch_two(
        INITIALIZER_CONFIG_PATH, GLOBAL_CONFIG_PATH
    )
    runner = MechanisticRunner(seip_ode)
    initial_state = initializer.get_initial_state()
    inferer = SMHInferer(
        GLOBAL_CONFIG_PATH, INFERER_CONFIG_PATH, runner, initial_state
    )

    return (inferer, runner)


def retrieve_post_samp(state):
    json_file = os.path.join(PREV_EPOCH_PATH, state, "checkpoint.json")
    post_samp = json.load(open(json_file, "r"))
    fitted_medians = {
        k: jnp.median(jnp.array(v), axis=(0, 1)) for k, v in post_samp.items()
    }

    return post_samp, fitted_medians


def replace_and_simulate(inferer, runner, particle):
    m = copy.deepcopy(inferer)
    m.config.INTRODUCTION_TIMES = [
        particle["INTRODUCTION_TIMES_0"],
        particle["INTRODUCTION_TIMES_1"],
    ]

    m.config.STRAIN_R0s = jnp.array(
        [
            particle["STRAIN_R0s_0"],
            particle["STRAIN_R0s_1"],
            particle["STRAIN_R0s_2"],
        ]
    )
    m.config.STRAIN_INTERACTIONS = jnp.array(
        [
            [particle["STRAIN_INTERACTIONS_0"], 1.0, 1.0],
            [particle["STRAIN_INTERACTIONS_3"], 1.0, 1.0],
            [
                particle["STRAIN_INTERACTIONS_6"],
                particle["STRAIN_INTERACTIONS_7"],
                1.0,
            ],
        ]
    )
    m.config.MIN_HOMOLOGOUS_IMMUNITY = particle["MIN_HOMOLOGOUS_IMMUNITY"]
    m.config.SEASONALITY_AMPLITUDE = particle["SEASONALITY_AMPLITUDE"]
    # m.config.SEASONALITY_SECOND_WAVE = particle[
    #     "SEASONALITY_SECOND_WAVE"
    # ]
    m.config.SEASONALITY_SHIFT = particle["SEASONALITY_SHIFT"]
    # set this to 1.0 to avoid sampling it
    m.config.INITIAL_INFECTIONS_SCALE = 1.0
    parameters = m.get_parameters()
    # we are sampling initial_infections too, so replace initial state with that posterior
    initial_state = m.scale_initial_infections(
        particle["INITIAL_INFECTIONS_SCALE"]
    )
    output = runner.run(
        initial_state,
        parameters,
        tf=EPOCH_2_DAYS,
    )

    return output


def sample_distributions_with_rng(parameters, rng):
    """
    an adapted version of sample_if_distribution within the inferer
    this version does not use numpyro to sample, instead replacing the value directly
    """
    for key, param in parameters.items():
        rng = get_new_seed(rng)
        # if distribution, sample and replace
        if issubclass(type(param), Dist.Distribution):
            param = float(param.sample(rng))
        # if list, check for distributions within and replace them
        elif isinstance(param, (np.ndarray, list)):
            param = np.array(param)  # cast np.array so we get .shape
            flat_param = np.ravel(param)  # Flatten the parameter array
            # check for distributions inside of the flattened parameter list
            if any(
                [
                    issubclass(type(param_lst), Dist.Distribution)
                    for param_lst in flat_param
                ]
            ):
                np.random.seed(rng[0])
                keys = np.random.randint(0, 10000, len(flat_param))
                # if we find distributions, sample them, then reshape back to the original shape
                flat_param = np.array(
                    [
                        (
                            float(
                                param_lst.sample(jax.random.PRNGKey(keys[i]))
                            )
                            if issubclass(type(param_lst), Dist.Distribution)
                            else param_lst
                        )
                        for i, param_lst in enumerate(flat_param)
                    ]
                )
                param = np.reshape(flat_param, param.shape)
        # else static param, do nothing
        parameters[key] = param
    return parameters


def replace_and_simulate_projection(projector, particle, runner, rng):
    m = copy.deepcopy(projector)
    # replace our distributions with a sample using the random number generator rng
    # caculate the mean and sd of the particle r0s to inform the projection r0s
    mean_r0 = np.mean(
        np.array([particle["STRAIN_R0s_%s" % i] for i in range(3)])
    )
    sd_r0 = np.std(np.array([particle["STRAIN_R0s_%s" % i] for i in range(3)]))
    for i in range(3, m.config.NUM_STRAINS):
        m.config.STRAIN_R0s[i].base_dist.loc = mean_r0
        m.config.STRAIN_R0s[i].base_dist.scale = sd_r0
    m.config.__dict__ = sample_distributions_with_rng(m.config.__dict__, rng)

    # we only keep one posterior for the r0 since we collapsed all strains together, lets use JN1
    m.config.STRAIN_R0s[0] = particle["STRAIN_R0s_0"]
    m.config.STRAIN_R0s[1] = particle["STRAIN_R0s_1"]
    m.config.STRAIN_R0s[2] = particle["STRAIN_R0s_2"]
    # use a weighted average of strain interactions to set the homologous initial immmunity for jn1
    m.config.STRAIN_INTERACTIONS[0][0] = particle["STRAIN_INTERACTIONS_0"]
    m.config.STRAIN_INTERACTIONS[1][0] = particle["STRAIN_INTERACTIONS_3"]
    m.config.STRAIN_INTERACTIONS[2][0] = particle["STRAIN_INTERACTIONS_6"]
    m.config.STRAIN_INTERACTIONS[2][1] = particle["STRAIN_INTERACTIONS_7"]
    m.config.MIN_HOMOLOGOUS_IMMUNITY = particle["MIN_HOMOLOGOUS_IMMUNITY"]
    m.config.SEASONALITY_AMPLITUDE = particle["SEASONALITY_AMPLITUDE"]
    m.config.SEASONALITY_SHIFT = particle["SEASONALITY_SHIFT"]
    parameters = m.get_parameters()
    output = runner.run(
        m.INITIAL_STATE,
        parameters,
        tf=PROJECTION_DAYS,
    )
    immunity_strain = utils.get_immunity(m, output)
    r0s_and_intro_times = {
        "STRAIN_R0s": m.config.STRAIN_R0s.tolist(),
        "INTRODUCTION_TIMES": m.config.INTRODUCTION_TIMES.tolist(),
    }

    return output, r0s_and_intro_times, immunity_strain


def simulate_hospitalization_history(
    output, ihr, ihr_immune_mult, ihr_jn1_mult, pop_sizes
):
    model_incidence = jnp.diff(output.ys[3], axis=0)

    model_incidence_no_exposures_non_jn1 = jnp.sum(
        model_incidence[:, :, 0, 0, :2], axis=-1
    )
    model_incidence_no_exposures_jn1 = model_incidence[:, :, 0, 0, 2]
    model_incidence_all_non_jn1 = jnp.sum(
        model_incidence[:, :, :, :, :2], axis=(2, 3, 4)
    )
    model_incidence_all_jn1 = jnp.sum(
        model_incidence[:, :, :, :, 2], axis=(2, 3)
    )
    model_incidence_w_exposures_non_jn1 = (
        model_incidence_all_non_jn1 - model_incidence_no_exposures_non_jn1
    )
    model_incidence_w_exposures_jn1 = (
        model_incidence_all_jn1 - model_incidence_no_exposures_jn1
    )

    model_hosps = (
        model_incidence_no_exposures_non_jn1 * ihr
        + model_incidence_no_exposures_jn1 * ihr * ihr_jn1_mult
        + model_incidence_w_exposures_non_jn1 * ihr * ihr_immune_mult
        + model_incidence_w_exposures_jn1
        * ihr
        * ihr_immune_mult
        * ihr_jn1_mult
    )

    # convert to 100k rate
    model_hosps = model_hosps / pop_sizes * 100000
    return model_hosps


def simulate_hospitalization(output, ihr, ihr_immune_mult, pop_sizes):
    model_incidence = jnp.sum(output.ys[3], axis=4)

    model_incidence_no_exposures = jnp.diff(
        model_incidence[:, :, 0, 0], axis=0
    )

    model_incidence_prev_exposure = jnp.sum(model_incidence, axis=(2, 3))
    model_incidence_prev_exposure = jnp.diff(
        model_incidence_prev_exposure, axis=0
    )
    # subtract no exposures to get prev_exposures
    model_incidence_prev_exposure -= model_incidence_no_exposures

    # calculate weekly model hospitalizations with the two IHRs we created
    model_hosps = (
        model_incidence_no_exposures * ihr
        + model_incidence_prev_exposure * ihr * ihr_immune_mult
    )
    # convert to 100k rate
    model_hosps = model_hosps / pop_sizes * 100000

    return model_hosps


def build_initial_state(particle):
    state_mapping, strain_mapping = utils.combined_strains_mapping(1, 0, 3)
    new_initial_state = []
    for compartment_name in ["s", "e", "i"]:
        # if S compartment, the last axis is not `strain`, which is important for combining
        strain_axis = compartment_name != "s"
        compartment_combined = np.array(
            particle["final_timestep_" + compartment_name]
        )

        if strain_axis:  # dont forget to pad strains for E and I
            compartment_combined = np.pad(
                compartment_combined,
                [
                    (0, 0),
                    (0, 120),
                    (0, 1),
                    (0, 4),
                ],
            )
        else:
            compartment_combined = np.pad(
                compartment_combined,
                [
                    (0, 0),
                    (0, 120),
                    (0, 1),
                    (0, 0),
                ],
            )
        new_initial_state.append(compartment_combined)
        # dont forget the C compartment which is all zeros
    new_initial_state.append(np.zeros(new_initial_state[-1].shape))
    return tuple(new_initial_state)


def project_and_dump_particle(idx, fitted_sample, inferer, runner):
    output_dict = {}
    history_solution = replace_and_simulate(inferer, runner, fitted_sample)
    scenario_config = os.path.join(EXP_PATH, "%s/%s.json" % (state, scenario))
    scenario_global = os.path.join(
        EXP_PATH, "%s/config_global_template.json" % (state)
    )
    # with the fitted samples, build the initial state
    spoof_final_states = {
        "final_timestep_%s" % c: history_solution.ys[i][-1]
        for i, c in zip(list(range(4)), ["s", "e", "i", "c"])
    }
    initial_state_particle = build_initial_state(spoof_final_states)
    rng = jax.random.PRNGKey((idx[0] + 1) * idx[1])

    # print("Projecting...")
    # create the static parameters object
    scenario_projector = SMHInferer(
        scenario_global, scenario_config, runner, initial_state_particle
    )
    output, r0s_and_intro_times, immunity_strain = (
        replace_and_simulate_projection(
            scenario_projector, fitted_sample, runner, rng
        )
    )
    # print("Done Projecting...")
    print(jnp.sum(output.ys[0][-1], axis=(1, 3)))
    output_list = output.ys[scenario_projector.config.COMPARTMENT_IDX.C]
    output_list_select_weeks = np.arange(len(output_list)) % 7
    output_list = output_list[output_list_select_weeks == 0, ...]
    new_cshape = list(output_list.shape)
    new_cshape[2] = 2
    new_output_list = np.zeros(tuple(new_cshape))
    new_output_list[:, :, 0, :, :] = output_list[:, :, 0, ...]
    new_output_list[:, :, 1, :, :] = np.sum(output_list[:, :, 1:, ...], axis=2)

    # store the sampled r0s and intro times of new strains
    # and the processed C compartment
    output_dict = r0s_and_intro_times
    output_dict["INIT_DATE"] = str(scenario_projector.config.INIT_DATE)
    output_dict["VACCINATION_RATES"] = utils.get_vaccination_rates(
        scenario_projector, PROJECTION_DAYS
    )
    output_dict["IMMUNITY_STRAIN"] = immunity_strain.tolist()
    output_dict["SEROPREVALENCE"] = utils.get_seroprevalence(
        scenario_projector, output
    ).tolist()
    output_dict["VARIANT_PROPORTIONS"] = utils.get_var_proportions(
        scenario_projector, output
    ).tolist()
    output_dict[str(idx)] = new_output_list.tolist()
    f = open(
        os.path.join(
            save_path,
            "c_compartment_%s_%s_%s.json" % (scenario, idx[0], idx[1]),
        ),
        "w",
    )
    json.dump(output_dict, f)
    f.close()

    return


def process_scenario(input_tuple, particle, save_path):
    """
    a function designed to run in parallel to execute 1 scenario on 1 state
    loads `NUM_PARTICLES_PER_CHAIN * 4` particles and executes the scenario defined by `scenario` json
    """
    # need to pass 1 param for pmap to work, so we do it this way
    # we will be building this up with info and returning it
    state = input_tuple[0]
    scenario = input_tuple[1]
    scenario_description = {"scenario": scenario}
    samp, _ = retrieve_post_samp(state)
    # nsamp = len(samp["ihr_0"][0])
    nchain = len(samp["ihr_0"])
    # magic 3 floating here, is this number of particles per chain for this state?

    fitted_samples = [
        {k: v[c][particle] for k, v in samp.items()} for c in range(nchain)
    ]
    chain_and_particle_vals = [(c, particle) for c in range(nchain)]
    (inferer, runner) = retrieve_inferer_obs(state)

    for idx, f in zip(chain_and_particle_vals, fitted_samples):
        print("running particle " + str(idx))
        project_and_dump_particle(idx, f, inferer, runner)

    return scenario_description


parser = argparse.ArgumentParser()
parser.add_argument(
    "-s",
    "--state",
    type=str,
    help="directory for the state to run, resembles USPS code of the state",
)

parser.add_argument(
    "-j", "--jobid", type=str, help="job-id of the state being run on Azure"
)

parser.add_argument("-sc", "--scenario", type=str, help="scenario name")
parser.add_argument("-p", "--particle", type=str, help="particle id")

args = parser.parse_args()
state = args.state
jobid = args.jobid
particle = int(args.particle)
scenario = args.scenario
save_path = "/output/projections/%s/%s/%s" % (jobid, state, scenario)
if not os.path.exists(save_path):
    os.makedirs(save_path, exist_ok=True)
scenario_description = process_scenario((state, scenario), particle, save_path)
