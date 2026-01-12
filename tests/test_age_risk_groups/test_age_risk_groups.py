import jax.numpy as jnp
import pytest

from dynode.config import Dimension
from dynode.config.bins import AgeBin, Bin
from examples.sir_age_risk_stratified import get_config, get_odeparams

## check shape ##

pytestmark = pytest.mark.parametrize(
    "inputs, expected_init_prop_shape, expected_cm_shape",
    [
        (
            {
                "age_demographics": jnp.array([1.0]),
                "risk_prop": jnp.array([[1.0]]),
                "age_contact_matrix": jnp.array([[1.0]]),
                "risk_contact_matrix": jnp.array([[1.0]]),
                "age_dimension": Dimension(
                    name="age", bins=[AgeBin(0, 99, "all")]
                ),
                "risk_dimension": Dimension(
                    name="risk", bins=[Bin(name="all")]
                ),
            },
            (1, 1),
            (1, 1, 1, 1),
        ),
        (
            {
                "age_demographics": jnp.array([0.7, 0.2, 0.1]),
                "risk_prop": jnp.array([[1.0], [1.0], [1.0]]),
                "age_contact_matrix": jnp.array(
                    [[0.7, 0.2, 0.1], [0.2, 0.7, 0.1], [0.1, 0.1, 0.8]]
                ),
                "risk_contact_matrix": jnp.array([[1.0]]),
                "age_dimension": Dimension(
                    name="age",
                    bins=[
                        AgeBin(0, 17, "young"),
                        AgeBin(18, 49, "adult"),
                        AgeBin(50, 99, "elderly"),
                    ],
                ),
                "risk_dimension": Dimension(
                    name="risk", bins=[Bin(name="all")]
                ),
            },
            (3, 1),
            (3, 1, 3, 1),
        ),
        (
            {
                "age_demographics": jnp.array([0.7, 0.2, 0.1]),
                "risk_prop": jnp.array([[0.1, 0.9], [0.6, 0.4], [0.8, 0.2]]),
                "age_contact_matrix": jnp.array(
                    [[0.7, 0.2, 0.1], [0.2, 0.7, 0.1], [0.1, 0.1, 0.8]]
                ),
                "risk_contact_matrix": jnp.array([[0.7, 0.3], [0.3, 0.7]]),
                "age_dimension": Dimension(
                    name="age",
                    bins=[
                        AgeBin(0, 17, "young"),
                        AgeBin(18, 49, "adult"),
                        AgeBin(50, 99, "elderly"),
                    ],
                ),
                "risk_dimension": Dimension(
                    name="risk", bins=[Bin(name="high"), Bin(name="low")]
                ),
            },
            (3, 2),
            (3, 2, 3, 2),
        ),
    ],
)


@pytest.fixture
def config_params():
    config_params = {
        "r_0": 2.0,
        "infectious_period": 7.0,
    }

    return config_params


def test_shape(
    config_params, inputs, expected_init_prop_shape, expected_cm_shape
):
    config = get_config(config_params | inputs)

    assert (
        config.initializer.get_initial_state()[0].shape
        == expected_init_prop_shape
    )
    assert get_odeparams(config).contact_matrix.shape == expected_cm_shape
