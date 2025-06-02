import numpyro.distributions as dist
import pytest
from pydantic import ValidationError

import dynode.config as config


def test_strain():
    """Test that the Strain config is valid."""
    try:
        strain = config.Strain(
            strain_name="valid_strain", r0=2, infectious_period=5
        )
    except ValidationError as e:
        pytest.fail(f"Strain raised ValidationError unexpectedly: {e}")

    assert strain.strain_name == "valid_strain"
    assert strain.r0 == 2
    assert strain.infectious_period == 5


# TODO: tests like this fail for now because we have not implemented complex validators
# capable of checking distribution supports in the case that the r0 is a distribution.
# def test_invalid_strain_r0():
#     """Test that the Strain config raises an error for invalid r0."""
#     with pytest.raises(ValidationError):
#         config.Strain(strain_name="invalid_strain", r0=-1, infectious_period=5)


def test_valid_strain_r0_distribution():
    """Test that the Strain config accepts a valid r0 distribution."""
    try:
        strain = config.Strain(
            strain_name="valid_strain_dist",
            r0=dist.Uniform(1.0, 3.0),
            infectious_period=5,
        )
    except ValidationError as e:
        pytest.fail(f"Strain with r0 distribution raised ValidationError: {e}")

    assert strain.strain_name == "valid_strain_dist"
    assert isinstance(strain.r0, dist.Distribution)
    assert strain.infectious_period == 5


def test_valid_introduced_strain():
    """Test that the IntroducedStrain config is valid."""
    try:
        introduced_strain = config.Strain(
            strain_name="introduced_strain",
            r0=2,
            infectious_period=5,
            is_introduced=True,
            introduction_time=100,
            introduction_percentage=0.1,
            introduction_scale=10,
            introduction_ages=[config.AgeBin(min_value=0, max_value=10)],
        )
    except ValidationError as e:
        pytest.fail(
            f"IntroducedStrain raised ValidationError unexpectedly: {e}"
        )

    assert introduced_strain.strain_name == "introduced_strain"
    assert introduced_strain.is_introduced is True
    assert introduced_strain.introduction_time == 100
    assert introduced_strain.introduction_percentage == 0.1
    assert introduced_strain.introduction_scale == 10
    assert introduced_strain.introduction_ages == [
        config.AgeBin(min_value=0, max_value=10)
    ]


# TODO: tests like this fail for now because we have not implemented the validators to check
# correlated fields like introduction_time, introduction_percentage, etc.
# def test_partial_introduced_strain():
#     """Attempt to create a strain that is introduced, but does not have all required fields."""
#     with pytest.raises(ValidationError):
#         config.Strain(
#             strain_name="partial_introduced_strain",
#             r0=2,
#             infectious_period=5,
#             is_introduced=True,
#             introduction_time=100,
#             introduction_percentage=0.1,
#             # Missing introduction_scale and introduction_ages
#         )
