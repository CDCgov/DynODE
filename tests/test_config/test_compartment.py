import jax.numpy as jnp
import pytest
from pydantic import ValidationError

import dynode.config as config


def test_compartment_valid():
    """Test that the Compartment config is valid."""
    try:
        compartment = config.Compartment(
            name="valid_compartment",
            dimensions=[
                config.Dimension(name="dim1", bins=[config.Bin(name="bin1")])
            ],
        )
    except Exception as e:
        pytest.fail(f"Compartment raised an exception unexpectedly: {e}")

    assert compartment.name == "valid_compartment"
    assert compartment.shape == (1,)
    assert compartment.values == jnp.zeros((1,))
    assert compartment.idx.dim1 == 0
    assert compartment.idx.dim1.bin1 == 0


def test_compartments_equal():
    """Test that two compartments with the same name and dimensions are equal."""
    compartment1 = config.Compartment(
        name="compartment1",
        dimensions=[
            config.Dimension(name="dim1", bins=[config.Bin(name="bin1")])
        ],
    )
    compartment2 = config.Compartment(
        name="compartment1",
        dimensions=[
            config.Dimension(name="dim1", bins=[config.Bin(name="bin1")])
        ],
    )
    assert (
        compartment1 == compartment2
    ), "Compartments with same name and dimensions should be equal"


def test_compartments_different_names():
    """Test that two compartments with different names are not equal."""
    compartment1 = config.Compartment(
        name="compartment1",
        dimensions=[
            config.Dimension(name="dim1", bins=[config.Bin(name="bin1")])
        ],
    )
    compartment2 = config.Compartment(
        name="compartment2",
        dimensions=[
            config.Dimension(name="dim1", bins=[config.Bin(name="bin1")])
        ],
    )
    assert (
        compartment1 != compartment2
    ), "Compartments with different names should not be equal"


def test_compartments_different_dimensions():
    """Test that two compartments with different dimensions are not equal."""
    compartment1 = config.Compartment(
        name="compartment1",
        dimensions=[
            config.Dimension(name="dim1", bins=[config.Bin(name="bin1")])
        ],
    )
    compartment2 = config.Compartment(
        name="compartment1",
        dimensions=[
            config.Dimension(name="dim2", bins=[config.Bin(name="bin2")])
        ],
    )
    assert (
        compartment1 != compartment2
    ), "Compartments with different dimensions should not be equal"


def test_compartment_invalid_dimensions():
    """Test that the Compartment config raises an error for invalid dimensions."""
    with pytest.raises(ValidationError):
        config.Compartment(
            name="invalid_compartment",
            dimensions=[
                config.Dimension(name="dim1", bins=[config.Bin(name="bin1")]),
                # Duplicate dimension name
                config.Dimension(name="dim1", bins=[config.Bin(name="bin2")]),
            ],
        )
