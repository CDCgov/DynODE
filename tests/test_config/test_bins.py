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


def test_wane_bin():
    """Test that the WaneBin config is valid."""
    try:
        wane_bin = config.WaneBin(
            name="wane_bin", waiting_time=10.0, base_protection=0.8
        )
    except ValidationError:
        pytest.fail("WaneBin raised ValidationError unexpectedly!")

    assert wane_bin.name == "wane_bin"
    assert wane_bin.waiting_time == 10.0
    assert wane_bin.base_protection == 0.8


def test_invalid_wane_bin_fields():
    """Test that the WaneBin config raises an error for invalid base_protection or waiting_time."""
    with pytest.raises(ValidationError):
        config.WaneBin(
            name="invalid_wane_bin", waiting_time=10.0, base_protection=1.2
        )

    with pytest.raises(ValidationError):
        config.WaneBin(
            name="invalid_wane_bin", waiting_time=-5.0, base_protection=0.8
        )
