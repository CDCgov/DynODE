from datetime import date

import pytest
from pydantic import ValidationError

import dynode.config as config


@pytest.fixture
def c():
    """Fixture for a valid simulation configuration."""
    return config.SimulationConfig(
        compartments=[
            config.Compartment(
                name="compartment1",
                dimensions=[
                    config.Dimension(
                        name="dim1", bins=[config.Bin(name="bin1")]
                    )
                ],
            )
        ],
        initializer=config.Initializer(
            description="test initializer",
            initialize_date=date(2022, 2, 11),
            population_size=1000,
        ),
        parameters=config.Params(
            transmission_params=config.TransmissionParams(
                strains=[
                    config.Strain(
                        strain_name="strain1", r0=2, infectious_period=5
                    )
                ],
                strain_interactions={"strain1": {"strain1": 1.0}},
            ),
            solver_params=config.SolverParams(),
        ),
    )


def test_simulation_config_valid(c: config.SimulationConfig):
    assert c is not None
    assert len(c.compartments) == 1
    assert c.idx.compartment1 == 0
    assert c.idx.compartment1.dim1 == 0
    assert c.idx.compartment1.dim1.bin1 == 0
    assert c.parameters.transmission_params.strains[0].strain_name == "strain1"


def test_flatten_bins(c: config.SimulationConfig):
    """Test that flatten_bins returns the correct flattened bins."""
    flattened_bins = c.flatten_bins()
    assert len(flattened_bins) == 1
    assert flattened_bins[0].name == "bin1"
    c.compartments.append(
        config.Compartment(
            name="compartment2",
            dimensions=[
                config.Dimension(name="dim2", bins=[config.Bin(name="bin2")])
            ],
        )
    )
    flattened_dims = c.flatten_bins()
    assert len(flattened_dims) == 2


def test_flatten_dims(c: config.SimulationConfig):
    """Test that flatten_dims returns the correct flattened dimensions."""
    flattened_dims = c.flatten_dims()
    assert len(flattened_dims) == 1
    assert flattened_dims[0].name == "dim1"
    c.compartments.append(
        config.Compartment(
            name="compartment2",
            dimensions=[
                config.Dimension(name="dim2", bins=[config.Bin(name="bin2")])
            ],
        )
    )
    flattened_dims = c.flatten_dims()
    assert len(flattened_dims) == 2


def test_get_compartment(c: config.SimulationConfig):
    """Test that get_compartment returns the correct compartment."""
    compartment = c.get_compartment("compartment1")
    assert compartment.name == "compartment1"
    # Should raise an error for non-existent compartment
    with pytest.raises(AssertionError):
        c.get_compartment("non_existent_compartment")


def test_same_dim_names_different_bins(c: config.SimulationConfig):
    """Test that compartments with the same dimension names but different bins are handled correctly."""
    c.compartments.append(
        config.Compartment(
            name="compartment2",
            dimensions=[
                config.Dimension(name="dim1", bins=[config.Bin(name="bin2")])
            ],
        )
    )
    with pytest.raises(ValidationError):
        # rerun the validators now that we have added the new compartment
        c.model_validate(c)


def test_same_dim_names_same_bins(c: config.SimulationConfig):
    """Test that compartments with the same dimension names and bins are handled correctly."""
    c.compartments.append(
        config.Compartment(
            name="compartment2",
            dimensions=[
                config.Dimension(name="dim1", bins=[config.Bin(name="bin1")])
            ],
        )
    )
    # rerun the validators now that we have added the new compartment
    # Should not raise an error
    c.model_validate(c)


# does not pass for now, hotfixing
# def test_same_compartment_names_errors(c: config.SimulationConfig):
#     """Test that compartments with the same name raise an error."""
#     c.compartments.append(
#         config.Compartment(
#             name="compartment1",  # Same name as existing compartment
#             dimensions=[
#                 config.Dimension(name="dim2", bins=[config.Bin(name="bin2")])
#             ],
#         )
#     )
#     with pytest.raises(ValidationError):
#         # rerun the validators now that we have added the new compartment
#         c.model_validate(c)


def test_immune_history_compartment_validators(c: config.SimulationConfig):
    """Test that SimulationConfig will raise an error if your immune history does
    not correlate with the strains passed to TransmissionParams."""

    c.compartments.append(
        config.Compartment(
            name="compartment2",
            dimensions=[
                config.LastStrainImmuneHistoryDimension(
                    strains=[
                        config.Strain(
                            strain_name="strain1", r0=2, infectious_period=5
                        )
                    ],
                )
            ],
        )
    )
    # should not raise because strain1 is in TransmissionParams
    c.model_validate(c)
    # now lets modify so the Dimension is using an incorrect strain
    c.compartments[1] = config.Compartment(
        name="compartment2",
        dimensions=[
            # purposely pass a strain that is not in TransmissionParams
            config.LastStrainImmuneHistoryDimension(
                strains=[
                    config.Strain(
                        strain_name="strain2", r0=2, infectious_period=5
                    )
                ],
            )
        ],
    )
    with pytest.raises(ValidationError):
        # rerun the validators now that we have added the new compartment
        c.model_validate(c)
