import numpyro
import pytest
from numpyro.handlers import substitute

import dynode.config as config


def sample_placeholder():
    return numpyro.sample("sample", config.PlaceholderSample())


def test_placeholder_sample_create():
    """Test that the PlaceholderSample config is valid."""
    try:
        config.PlaceholderSample()
    except Exception as e:
        pytest.fail(f"PlaceholderSample raised Exception unexpectedly: {e}")


def test_placeholder_sample_errors_when_sampled_directly():
    """Test that the PlaceholderSample raises an error when sampled directly."""
    with pytest.raises(config.SamplePlaceholderError):
        sample_placeholder()


def test_placeholder_sample_substitution():
    """Test that the PlaceholderSample can be substituted with a real sample."""
    sample_with_substitute = substitute(
        sample_placeholder, data={"sample": 42}
    )
    try:
        assert (
            sample_with_substitute() == 42
        ), "Substituted sample did not return expected value"
    except config.SamplePlaceholderError as e:
        pytest.fail(f"Substituted sample raised SamplePlaceholderError: {e}")
