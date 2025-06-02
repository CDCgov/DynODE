import pytest
from pydantic import ValidationError

import dynode.config as config


def test_transmission_params():
    """Test that the TransmissionParams config is valid."""
    strains = [
        config.Strain(strain_name="strain1", r0=2, infectious_period=5),
        config.Strain(strain_name="strain2", r0=2, infectious_period=5),
    ]
    strain_interactions = {
        "strain1": {"strain1": 1.0, "strain2": 0.5},
        "strain2": {"strain1": 0.5, "strain2": 1.0},
    }
    try:
        transmission_params = config.TransmissionParams(
            strains=strains, strain_interactions=strain_interactions
        )
    except ValidationError as e:
        pytest.fail(
            f"TransmissionParams raised ValidationError unexpectedly: {e}"
        )

    assert transmission_params.strains == strains
    assert transmission_params.strain_interactions == strain_interactions


def test_transmission_params_invalid_strain_interactions():
    """Test that the TransmissionParams raises an error for invalid strain interactions."""
    strains = [
        config.Strain(strain_name="strain1", r0=2, infectious_period=5),
        config.Strain(strain_name="strain2", r0=2, infectious_period=5),
    ]
    strain_interactions = {
        "strain1": {"strain1": 1.0, "strain2": 0.5},
        "strain2": {"strain1": 0.5},  # Missing interaction with strain2
    }
    with pytest.raises(ValidationError):
        config.TransmissionParams(
            strains=strains, strain_interactions=strain_interactions
        )
    # Extra strain in interactions not in strains, TODO have this raise
    # strain_interactions = {
    #     "strain1": {"strain1": 1.0, "strain2": 0.5, "strain3": 0.5},
    #     "strain2": {"strain1": 0.5, "strain2": 1.0, "strain3": 0.5},
    #     "strain3": {"strain1": 0.5, "strain2": 0.5, "strain3": 1.0},
    # }
    # with pytest.raises(ValidationError):
    #     config.TransmissionParams(
    #         strains=strains, strain_interactions=strain_interactions
    #     )
