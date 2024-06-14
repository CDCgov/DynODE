import jax
import jax.numpy as jnp
import numpy as np
import pytest

from config.config import Config
from mechanistic_model.mechanistic_inferer import MechanisticInferer
from mechanistic_model.mechanistic_runner import MechanisticRunner
from mechanistic_model.static_value_parameters import StaticValueParameters
from model_odes.seip_model import seip_ode

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
    inferer_posteriors = inferer.load_posterior_particle(particle_num=0)
    for chain in range(inferer.config.INFERENCE_NUM_CHAINS):
        individual_particle = inferer_posteriors[(0, chain)]
        for i, compartment in enumerate(individual_particle["solution"].ys):
            # did we reproduce the same timeline, the values will be different since this is a particle
            assert (
                compartment.shape == synthetic_solution.ys[i].shape
            ), "load_posterior_particle produced different timeline shapes than what was fit on"


def test_external_posteriors():
    inferer1_posteriors = inferer.load_posterior_particle(
        particle_num=0, tf=100
    )
    inferer2 = MechanisticInferer(
        GLOBAL_CONFIG_PATH, INFERER_CONFIG_PATH, runner, fake_initial_state
    )
    inferer2_posteriors = inferer2.load_posterior_particle(
        0, tf=100, external_posteriors=mc.get_samples(group_by_chain=True)
    )
    for chain in range(inferer.config.INFERENCE_NUM_CHAINS):
        inferer1_chain = inferer1_posteriors[(0, chain)]
        inferer2_chain = inferer2_posteriors[(0, chain)]
        # check that all posterios are the same including all lists
        assert np.isclose(
            inferer1_chain["solution"].ys[inferer.config.COMPARTMENT_IDX.C],
            inferer2_chain["solution"].ys[inferer2.config.COMPARTMENT_IDX.C],
        ).all(), "rerunning the same inference particle with two different inferers produces different output"
