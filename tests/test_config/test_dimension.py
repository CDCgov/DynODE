import math

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

    assert dimension.name == "vax"  # default
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


def test_fully_stratified_immune_history_dimension():
    """Test that the FullStratifiedImmuneHistoryDimension builder works correctly."""
    strains = [
        config.Strain(strain_name=f"s{i}", r0=2, infectious_period=2)
        for i in range(3)
    ]
    try:
        dimension = config.FullStratifiedImmuneHistoryDimension(
            strains=strains
        )
    except Exception as e:
        pytest.fail(
            f"FullStratifiedImmuneHistoryDimension raised an exception unexpectedly: {e}"
        )

    assert dimension.name == "hist"
    # all combinations of strains + "none"
    assert len(dimension.bins) == 2 ** len(strains)
    # all combinations of strains interactions
    expected_bins = [
        "none",
        "s0",
        "s1",
        "s2",
        "s0_s1",
        "s0_s2",
        "s1_s2",
        "s0_s1_s2",
    ]
    for expected_bin, hist_bin in zip(expected_bins, dimension.bins):
        assert hist_bin.name == expected_bin, "Expected bin name mismatch."


def test_invalid_fully_stratified_immune_history_dimension():
    """Test that the FullStratifiedImmuneHistoryDimension raises an error for same strains."""
    with pytest.raises(ValidationError):
        config.FullStratifiedImmuneHistoryDimension(
            strains=[
                config.Strain(
                    strain_name="same_strain", r0=2, infectious_period=2
                ),
                config.Strain(
                    strain_name="same_strain", r0=2, infectious_period=2
                ),
            ]
        )


def test_last_strain_in_immune_history_dimension():
    """Test that the LastStrainImmuneHistoryDimension produces the desired bins."""
    strains = [
        config.Strain(strain_name="s1", r0=2, infectious_period=2),
        config.Strain(strain_name="s2", r0=2, infectious_period=2),
        config.Strain(strain_name="s3", r0=2, infectious_period=2),
    ]
    try:
        dimension = config.LastStrainImmuneHistoryDimension(strains=strains)
    except Exception as e:
        pytest.fail(
            f"FullStratifiedImmuneHistoryDimension raised an exception unexpectedly: {e}"
        )

    assert dimension.name == "hist"
    assert len(dimension.bins) == len(strains) + 1  # +1 for "none"
    expected_bins = ["none", "s1", "s2", "s3"]
    for expected_bin, hist_bin in zip(expected_bins, dimension.bins):
        assert hist_bin.name == expected_bin, "Expected bin name mismatch."


def test_wane_dimension():
    """Test that the WaneDimension produces the desired bins."""
    waiting_times = [1.0, 2.0, 3.0, math.inf]
    base_protections = [0.5, 0.6, 0.7, 0.9]
    try:
        dimension = config.WaneDimension(
            waiting_times=waiting_times,
            base_protections=base_protections,
        )
    except Exception as e:
        pytest.fail(f"WaneDimension raised an exception unexpectedly: {e}")

    assert dimension.name == "wane"
    assert len(dimension.bins) == len(waiting_times)
    for i, bin in enumerate(dimension.bins):
        bin: config.WaneBin = bin  # type hint
        assert bin.waiting_time == waiting_times[i]
        assert bin.base_protection == base_protections[i]
        assert bin.name == f"W{i}"  # default


def test_wane_dimension_no_inf():
    """Test that the WaneDimension errors if not passed an math.inf as final wane time."""
    waiting_times = [1.0, 2.0, 3.0, 4.0]
    base_protections = [0.5, 0.6, 0.7, 0.9]
    with pytest.raises(ValidationError):
        config.WaneDimension(
            waiting_times=waiting_times,
            base_protections=base_protections,
        )
