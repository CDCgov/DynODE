import pytest
from pydantic import ValidationError

import dynode.config as config


def test_valid_dimension():
    """Test that the Dimension config is valid."""
    try:
        dimension = config.Dimension(
            name="valid_dimension", bins=[config.Bin(name="bin1")]
        )
    except Exception as e:
        pytest.fail(f"Dimension raised an exception unexpectedly: {e}")

    assert dimension.name == "valid_dimension"
    assert len(dimension.bins) == 1
    assert dimension.bins[0].name == "bin1"


def test_mixed_type_bins():
    """Test that Dimension raises an error for mixed type bins."""
    with pytest.raises(ValidationError):
        config.Dimension(
            name="mixed_type_dimension",
            bins=[
                config.Bin(name="bin1"),
                config.DiscretizedPositiveIntBin(min_value=0, max_value=10),
            ],
        )


def test_empty_bins():
    """Test that Dimension raises an error for empty bins."""
    with pytest.raises(ValidationError):
        config.Dimension(name="empty_bins_dimension", bins=[])


def test_non_unique_bin_names():
    """Test that Dimension raises an error for non-unique bin names."""
    with pytest.raises(ValidationError):
        config.Dimension(
            name="non_unique_dimension",
            bins=[
                config.Bin(name="bin1"),
                config.Bin(name="bin1"),  # Duplicate name
            ],
        )


def test_discretized_int_bins_sorted():
    """Test that Dimension with DiscretizedPositiveIntBin is sorted and has no gaps."""
    try:
        dimension = config.Dimension(
            name="sorted_dimension",
            bins=[
                config.DiscretizedPositiveIntBin(min_value=0, max_value=10),
                config.DiscretizedPositiveIntBin(min_value=11, max_value=20),
            ],
        )
    except Exception as e:
        pytest.fail(f"Dimension raised an exception unexpectedly: {e}")

    assert len(dimension.bins) == 2
    assert dimension.bins[0].min_value == 0
    assert dimension.bins[1].min_value == 11


def test_discretized_int_bins_overlap():
    """Test that Dimension with overlapping DiscretizedPositiveIntBin raises an error."""
    with pytest.raises(ValidationError):
        config.Dimension(
            name="overlapping_dimension",
            bins=[
                config.DiscretizedPositiveIntBin(min_value=0, max_value=10),
                config.DiscretizedPositiveIntBin(min_value=5, max_value=15),
            ],
        )


def test_discretized_int_bins_not_sorted():
    """Test that Dimension with DiscretizedPositiveIntBin not sorted raises an error."""
    with pytest.raises(ValidationError):
        config.Dimension(
            name="not_sorted_dimension",
            bins=[
                config.DiscretizedPositiveIntBin(min_value=11, max_value=20),
                config.DiscretizedPositiveIntBin(min_value=0, max_value=10),
            ],
        )


def test_discretized_int_bins_with_gaps():
    """Test that Dimension with DiscretizedPositiveIntBin with gaps raises an error."""
    with pytest.raises(ValidationError):
        config.Dimension(
            name="gaps_dimension",
            bins=[
                config.DiscretizedPositiveIntBin(min_value=0, max_value=10),
                config.DiscretizedPositiveIntBin(min_value=12, max_value=20),
            ],
        )


def test_vaccination_dimension():
    """Test that the VaccinationDimension builder works correctly."""
    num_shots = 2
    try:
        dimension = config.VaccinationDimension(
            max_ordinal_vaccinations=num_shots,
            seasonal_vaccination=False,
        )
    except Exception as e:
        pytest.fail(
            f"VaccinationDimension raised an exception unexpectedly: {e}"
        )

    assert dimension.name == "vax"
    assert len(dimension.bins) == num_shots + 1  # +1 for the 0-shot bin
    # assert dimension.seasonal_vaccination is False
    assert dimension.bins[0].min_value == 0
    assert dimension.bins[0].max_value == 0
    assert dimension.bins[1].min_value == 1
    assert dimension.bins[1].max_value == 1


def test_vaccination_dimension_seasonal():
    """Test that the VaccinationDimension builder works correctly."""
    num_shots = 2
    try:
        dimension = config.VaccinationDimension(
            max_ordinal_vaccinations=num_shots,
            seasonal_vaccination=True,
        )
    except Exception as e:
        pytest.fail(
            f"VaccinationDimension raised an exception unexpectedly: {e}"
        )

    assert dimension.name == "vax"
    assert len(dimension.bins) == num_shots + 2  # +1 for seasonal
    # assert dimension.seasonal_vaccination is True
    assert dimension.bins[0].min_value == 0
    assert dimension.bins[0].max_value == 0
    assert dimension.bins[1].min_value == 1
    assert dimension.bins[1].max_value == 1
    assert dimension.bins[2].min_value == 2
    assert dimension.bins[2].max_value == 2
    assert dimension.bins[3].min_value == 3
    assert dimension.bins[3].max_value == 3
