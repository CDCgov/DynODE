import numpy as np
import numpyro.distributions as dist
import pytest
from jax import Array
from jax.random import PRNGKey
from numpyro.handlers import trace
from pydantic import BaseModel, ConfigDict

from dynode.config import DeterministicParameter
from dynode.infer import (
    resolve_deterministic,
    sample_distributions,
    sample_then_resolve,
)


def test_sample_distributions():
    test = {"a": dist.Normal()}
    try:
        test = sample_distributions(test, rng_key=PRNGKey(0))
    except Exception as e:
        pytest.fail(f"Unexpected exception {e}")


def test_sample_distributions_objects():
    # build up a testing pydantic class that will hold the distribution
    class test(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)
        x: dist.Distribution | Array

    t = test(x=dist.Normal())
    try:
        t = sample_distributions(t, rng_key=PRNGKey(0))
    except Exception as e:
        pytest.fail(f"Unexpected exception {e}")
    assert isinstance(t.x, Array), "sampling was not successful in an object"


def test_sample_distributions_list():
    test = {"a": [1, dist.Normal()], "b": np.array([1, dist.Normal()])}
    try:
        test = sample_distributions(test, rng_key=PRNGKey(0))
    except Exception as e:
        pytest.fail(f"Unexpected exception {e}")
    assert isinstance(test["a"][1], Array)
    assert isinstance(test["b"][1], Array)


def test_sample_distributions_naming():
    test = {
        "a": dist.Normal(),
        "b": [1, dist.Normal()],
        "c": np.array([dist.Normal(), 1]),
        "d": {"nested_dict": dist.Normal()},
    }
    t = trace(
        lambda: sample_distributions(test, rng_key=PRNGKey(0))
    ).get_trace()
    assert "a" in t.keys(), (
        f"{test} samples not named correctly, expected 'a', got {t.keys()}"
    )
    assert "b_1" in t.keys(), (
        f"{test} samples not named correctly, expected 'b_1', got {t.keys()}"
    )
    assert "c_0" in t.keys(), (
        f"{test} samples not named correctly, expected 'c_0', got {t.keys()}"
    )
    assert "d_nested_dict" in t.keys(), (
        f"{test} samples not named correctly, expected 'd_nested_dict', got {t.keys()}"
    )


def test_sample_distributions_naming_prefix():
    test = {
        "a": dist.Normal(),
        "b": [1, dist.Normal()],
        "c": np.array([dist.Normal(), 1]),
        "d": {"nested_dict": dist.Normal()},
    }
    t = trace(
        lambda: sample_distributions(test, rng_key=PRNGKey(0), _prefix="test_")
    ).get_trace()
    assert "test_a" in t.keys(), (
        f"{test} samples not named correctly, expected 'a', got {t.keys()}"
    )
    assert "test_b_1" in t.keys(), (
        f"{test} samples not named correctly, expected 'b_1', got {t.keys()}"
    )
    assert "test_c_0" in t.keys(), (
        f"{test} samples not named correctly, expected 'c_0', got {t.keys()}"
    )
    assert "test_d_nested_dict" in t.keys(), (
        f"{test} samples not named correctly, expected 'd_nested_dict', got {t.keys()}"
    )


def test_resolve_deterministic():
    test = {"a": 1, "b": DeterministicParameter("a")}
    test = resolve_deterministic(test, root_params=test)
    assert test["b"] == 1, f"unable to resolve {test}"


def test_resolve_deterministic_lists():
    test = {"a": [0, 1], "b": DeterministicParameter("a", index=1)}
    test = resolve_deterministic(test, root_params=test)
    assert test["b"] == 1, f"unable to resolve {test}"


def test_resolve_deterministic_list_slices():
    test = {
        "a": [0, 1, 3, 4],
        "b": DeterministicParameter("a", index=slice(0, 2)),
    }
    test = resolve_deterministic(test, root_params=test)
    assert test["b"] == [0, 1], f"unable to resolve {test}"


def test_sample_then_resolve():
    test = {"a": dist.Normal(), "b": DeterministicParameter("a")}
    test = sample_then_resolve(test, rng_key=PRNGKey(0))
    assert isinstance(test["a"], Array)
    assert isinstance(test["b"], Array)
    assert test["a"] == test["b"]


def test_sample_then_resolve_naming_prefix():
    test = {
        "a": dist.Normal(),
        "b": [1, dist.Normal()],
        "c": np.array([dist.Normal(), 1]),
        "d": {"nested_dict": dist.Normal()},
        "e": DeterministicParameter("a"),
    }
    t = trace(
        lambda: sample_then_resolve(test, rng_key=PRNGKey(0), _prefix="test_"),
    ).get_trace()
    assert "test_a" in t.keys(), (
        f"{test} samples not named correctly, expected 'test_a', got {t.keys()}"
    )
    assert "test_b_1" in t.keys(), (
        f"{test} samples not named correctly, expected 'test_b_1', got {t.keys()}"
    )
    assert "test_c_0" in t.keys(), (
        f"{test} samples not named correctly, expected 'test_c_0', got {t.keys()}"
    )
    assert "test_d_nested_dict" in t.keys(), (
        f"{test} samples not named correctly, expected 'test_d_nested_dict', got {t.keys()}"
    )
    assert "test_e" in t.keys(), (
        f"{test} samples not named correctly, expected 'test_e', got {t.keys()}"
    )
