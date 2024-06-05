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

inferer = MechanisticInferer(
    GLOBAL_CONFIG_PATH, INFERER_CONFIG_PATH, runner, fake_initial_state
)


def test_no_memory_leaking():
    try:
        with jax.check_tracer_leaks():
            inferer.infer(synthetic_hosp_obs)
    except Exception as e:
        pytest.fail(
            "jax.check_tracer_leaks raised %s A memory leak likely occured somewhere within the inferer!"
            % e
        )
