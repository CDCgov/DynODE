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
    strain_interactions = {
        "strain1": {"strain1": 1.0, "strain2": 0.5, "strain3": 0.5},
        "strain2": {"strain1": 0.5, "strain2": 1.0, "strain3": 0.5},
        "strain3": {"strain1": 0.5, "strain2": 0.5, "strain3": 1.0},
    }
    with pytest.raises(ValidationError):
        config.TransmissionParams(
            strains=strains, strain_interactions=strain_interactions
        )


def test_transmission_params_inconsistent_strains():
    """Test that the TransmissionParams raises an error for inconsistent strains."""
    strains = [
        config.Strain(
            strain_name="strain1",
            r0=2,
            infectious_period=5,
            exposed_to_infectious=5.0,
        ),
        config.Strain(
            strain_name="strain2", r0=2, infectious_period=5
        ),  # missing exposed_to_infectious
    ]
    strain_interactions = {
        "strain1": {"strain1": 1.0, "strain2": 0.5},
        "strain2": {"strain1": 0.5, "strain2": 1.0},
    }
    with pytest.raises(ValidationError):
        config.TransmissionParams(
            strains=strains, strain_interactions=strain_interactions
        )
    # test again but with different field missing
    strains = [
        config.Strain(
            strain_name="strain1",
            r0=2,
            infectious_period=5,
            vaccine_efficacy={0: 0.8, 1: 0.9},
        ),
        config.Strain(
            strain_name="strain2", r0=2, infectious_period=5
        ),  # missing vaccine_efficacy
    ]
    with pytest.raises(ValidationError):
        config.TransmissionParams(
            strains=strains, strain_interactions=strain_interactions
        )


def test_solver_params():
    """Test that the SolverParams config is valid."""
    try:
        solver_params = config.SolverParams(
            max_steps=1000,
            ode_solver_rel_tolerance=1e-6,
            ode_solver_abs_tolerance=1e-9,
        )
    except ValidationError as e:
        pytest.fail(f"SolverParams raised ValidationError unexpectedly: {e}")
    assert solver_params.max_steps == 1000
    assert solver_params.ode_solver_rel_tolerance == 1e-6
    assert solver_params.ode_solver_abs_tolerance == 1e-9


def test_solver_params_invalid():
    """Test that the SolverParams raises an error for invalid parameters."""
    with pytest.raises(ValidationError):
        config.SolverParams(
            max_steps=-1000,  # Invalid negative max_steps
            ode_solver_rel_tolerance=1e-6,
            ode_solver_abs_tolerance=1e-9,
        )
    with pytest.raises(ValidationError):
        config.SolverParams(
            max_steps=1000,
            ode_solver_rel_tolerance=-1e-6,  # Invalid negative tolerance
            ode_solver_abs_tolerance=1e-9,
        )
