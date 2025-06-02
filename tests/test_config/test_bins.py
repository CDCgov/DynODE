import pytest
from pydantic import ValidationError

import dynode.config as config


def test_descretized_int_bin():
    """Test that the bin config is valid."""
    try:
        int_bin = config.DiscretizedPositiveIntBin(min_value=0, max_value=10)
    except AssertionError:
        pytest.fail(
            "DiscretizedPositiveIntBin raised AssertionError unexpectedly!"
        )
    assert int_bin.min_value == 0
    assert int_bin.max_value == 10


def test_invalid_descretized_int_bin():
    """Test that the bin config raises an error for invalid min/max."""
    with pytest.raises(ValidationError):
        config.DiscretizedPositiveIntBin(min_value=10, max_value=0)


def test_dynode_name():
    """Test that the bin config raises an error for invalid name."""
    with pytest.raises(ValidationError):
        config.Bin(name="1_invalid_name")

    try:
        valid_bin = config.Bin(name="valid_name")
    except ValidationError:
        pytest.fail("Bin raised ValueError unexpectedly for valid name!")

    assert valid_bin.name == "valid_name"
