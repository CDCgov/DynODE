import pytest

import dynode.config as config


def test_deterministic_parameter():
    """Test that the DeterministicParameter resolves correctly."""
    param_state = {
        "base_param": [1, 2, 3],
        "another_param": 5,
        "dependent_param": config.DeterministicParameter(
            depends_on="base_param", index=1
        ),
        "dependent_param2": config.DeterministicParameter(
            depends_on="another_param"
        ),
    }

    resolved_value = param_state["dependent_param"].resolve(param_state)
    assert resolved_value == param_state["base_param"][1]
    assert (
        param_state["dependent_param2"].resolve(param_state)
        == param_state["another_param"]
    )


def test_invalid_deterministic_parameter():
    """Test that the DeterministicParameter raises an error for invalid index."""
    param_state = {
        "base_param": [1, 2, 3],
        "dependent_param": config.DeterministicParameter(
            depends_on="base_param", index=3
        ),
        "dependent_param_key_error": config.DeterministicParameter(
            depends_on="base_param", index=(1, 2)
        ),
        "dependent_param2": config.DeterministicParameter(
            depends_on="missing_param"
        ),
    }

    with pytest.raises(Exception):
        param_state["dependent_param"].resolve(param_state)
    with pytest.raises(Exception):
        param_state["dependent_param_key_error"].resolve(param_state)
    with pytest.raises(Exception):
        param_state["dependent_param2"].resolve(param_state)
