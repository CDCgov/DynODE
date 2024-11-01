import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dyn_ode import (
    Config,
    MechanisticInferer,
    MechanisticRunner,
    StaticValueParameters,
)
from dyn_ode.model_odes import seip_ode

runner = MechanisticRunner(seip_ode)
GLOBAL_CONFIG_PATH = "tests/test_config_global.json"
INFERER_CONFIG_PATH = "tests/test_config_inferer.json"
RUNNER_CONFIG_PATH = "tests/test_config_runner.json"
GLOBAL_JSON = open(GLOBAL_CONFIG_PATH, "r").read()
global_config = Config(GLOBAL_JSON)
S_SHAPE = (
    global_config.NUM_AGE_GROUPS,
    2**global_config.NUM_STRAINS,
    global_config.MAX_VACCINATION_COUNT + 1,
    global_config.NUM_WANING_COMPARTMENTS,
)
EIC_SHAPE = (
    global_config.NUM_AGE_GROUPS,
    2**global_config.NUM_STRAINS,
    global_config.MAX_VACCINATION_COUNT + 1,
    global_config.NUM_STRAINS,
)
fake_initial_state = (
    5000 * jnp.ones(S_SHAPE),  # S
    jnp.zeros(EIC_SHAPE),  # E
    100 * jnp.ones(EIC_SHAPE),  # I
    jnp.zeros(EIC_SHAPE),  # C
)
static_params = StaticValueParameters(
    fake_initial_state,
    RUNNER_CONFIG_PATH,
    GLOBAL_CONFIG_PATH,
)
synthetic_solution = runner.run(
    fake_initial_state,
    tf=100,
    args=static_params.get_parameters(),
)
ihr = [0.002, 0.004, 0.008, 0.06]
model_incidence = jnp.sum(synthetic_solution.ys[3], axis=(2, 3, 4))
model_incidence = jnp.diff(model_incidence, axis=0)
synthetic_hosp_obs = np.asarray(model_incidence) * ihr
synthetic_hosp_obs = jnp.rint(synthetic_hosp_obs).astype(int)

inferer = MechanisticInferer(
    GLOBAL_CONFIG_PATH, INFERER_CONFIG_PATH, runner, fake_initial_state
)

# test_no_memory_leaking is implicit, this way we can save the results and run other tests with them
# that means this test wont show up as a run test, but it will still run every time
try:
    with jax.check_tracer_leaks():
        mc = inferer.infer(synthetic_hosp_obs)
except Exception as e:
    pytest.fail(
        "jax.check_tracer_leaks raised %s A memory leak likely occured somewhere within the inferer!"
        % e
    )


def test_load_posterior_particle():
    # select particle 0 across chains
    load_across_chains = [
        (chain, 0) for chain in range(inferer.config.INFERENCE_NUM_CHAINS)
    ]
    inferer_posteriors = inferer.load_posterior_particle(load_across_chains)
    for chain in range(inferer.config.INFERENCE_NUM_CHAINS):
        individual_particle = inferer_posteriors[(chain, 0)]
        for i, compartment in enumerate(individual_particle["solution"].ys):
            # did we reproduce the same timeline, the values will be different since this is a particle
            assert (
                compartment.shape == synthetic_solution.ys[i].shape
            ), "load_posterior_particle produced different timeline shapes than what was fit on"


def test_external_posteriors():
    load_across_chains = [
        (chain, 0) for chain in range(inferer.config.INFERENCE_NUM_CHAINS)
    ]
    inferer1_posteriors = inferer.load_posterior_particle(
        load_across_chains, tf=100
    )
    inferer2 = MechanisticInferer(
        GLOBAL_CONFIG_PATH, INFERER_CONFIG_PATH, runner, fake_initial_state
    )
    inferer2_posteriors = inferer2.load_posterior_particle(
        load_across_chains,
        tf=100,
        external_particle=mc.get_samples(group_by_chain=True),
    )
    for chain in range(inferer.config.INFERENCE_NUM_CHAINS):
        inferer1_chain = inferer1_posteriors[(chain, 0)]
        inferer2_chain = inferer2_posteriors[(chain, 0)]
        # check that all posterios are the same including all lists
        assert np.isclose(
            inferer1_chain["solution"].ys[inferer.config.COMPARTMENT_IDX.C],
            inferer2_chain["solution"].ys[inferer2.config.COMPARTMENT_IDX.C],
        ).all(), "rerunning the same inference particle with two different inferers produces different output"


def test_random_sampling_across_chains_and_particles():
    # select particle 0 across chains
    load_across_chains = [
        (chain, 0) for chain in range(inferer.config.INFERENCE_NUM_CHAINS)
    ] + [(chain, 1) for chain in range(inferer.config.INFERENCE_NUM_CHAINS)]
    inferer1_posteriors = inferer.load_posterior_particle(
        load_across_chains, tf=100
    )
    inferer2 = MechanisticInferer(
        GLOBAL_CONFIG_PATH, INFERER_CONFIG_PATH, runner, fake_initial_state
    )
    # lets load external particles, but drop a single parameter from them
    # this will force the inferer to re-sample that parameter
    external_particles = mc.get_samples(group_by_chain=True)
    dropped_param_name = list(external_particles.keys())[0]
    print(external_particles[dropped_param_name])
    del external_particles[dropped_param_name]
    inferer2_posteriors = inferer2.load_posterior_particle(
        load_across_chains,
        tf=100,
        external_particle=external_particles,
    )
    for chain in range(inferer.config.INFERENCE_NUM_CHAINS):
        # get both particles across both inferers for this chain
        inferer1_chain_particle_0 = inferer1_posteriors[(chain, 0)]
        # inferer1_chain_particle_1 = inferer1_posteriors[(chain, 1)]
        inferer2_chain_particle_0 = inferer2_posteriors[(chain, 0)]
        inferer2_chain_particle_1 = inferer2_posteriors[(chain, 1)]
        # make sure that we are randomly sampling the dropped parameter
        # this means inferer1 and inferer2 should have different values for that parameter
        # even if it is executed within the same particle/chain
        # also make sure that the same chain diff particle also samples a different value
        assert (
            inferer1_chain_particle_0["parameters"][dropped_param_name]
            != inferer2_chain_particle_0["parameters"][dropped_param_name]
        ), "you are not correctly sampling parameters within `load_posterior_particle"
        # double check that different particles within the same posteriors also differ
        # TODO this play inference scenario fails because the sampler cant move anywhere with such weird initial conditions...
        # need to make a better testing of inference in general
        # assert (
        #     inferer1_chain_particle_0["parameters"][dropped_param_name]
        #     != inferer1_chain_particle_1["parameters"][dropped_param_name]
        # ), (
        #     "sampled two different particles, but got the same posterior value for %s"
        #     % dropped_param_name
        # )
        # check that the random sampling of "new" parameters does differ from particle to particle
        assert (
            inferer2_chain_particle_0["parameters"][dropped_param_name]
            != inferer2_chain_particle_1["parameters"][dropped_param_name]
        ), (
            "sampled a new parameter %s across particles but got the same values"
            % dropped_param_name
        )
    if inferer.config.INFERENCE_NUM_CHAINS >= 2:
        inferer2_chain_0_particle_0 = inferer2_posteriors[(0, 0)]
        inferer2_chain_1_particle_0 = inferer2_posteriors[(1, 0)]
        assert (
            inferer2_chain_0_particle_0["parameters"][dropped_param_name]
            != inferer2_chain_1_particle_0["parameters"][dropped_param_name]
        ), (
            "sampled a new parameter %s across chains but got the same values"
            % dropped_param_name
        )
    else:  # only have a single chain, cant run this test
        assert False, (
            "Unable to run all tests within test_random_sampling_across_chains_and_particles "
            "since you have only one chain! check test_config_inferer.json"
        )


def test_debug_inferer():
    """A simple test to make sure the _debug_likelihood function does not explode and correctly passes kwargs"""
    inferer._debug_likelihood(
        tf=len(synthetic_hosp_obs), obs_metrics=synthetic_hosp_obs
    )
