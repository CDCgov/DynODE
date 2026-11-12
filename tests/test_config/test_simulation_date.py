import os
from dataclasses import dataclass
from datetime import date

import pytest
from jax.random import PRNGKey
from numpyro.distributions import (
    Normal,
    TruncatedNormal,
)
from pydantic import BaseModel, ValidationError

import dynode.config as config


@pytest.fixture(autouse=True)
def setup_environment():
    # to make sure the tests dont impact eachother, lets always clear
    # the env flag before and after each test.
    if f"DYNODE_INITIALIZATION_DATE({os.getpid()})" in os.environ:
        del os.environ[f"DYNODE_INITIALIZATION_DATE({os.getpid()})"]


def test_simulation_date():
    """Test that the SimulationDate config is valid."""
    try:
        config.set_dynode_init_date_flag(date(2022, 2, 11))
        simulation_day = config.simulation_day(2022, 2, 11)
    except ValidationError as e:
        pytest.fail(f"SimulationDate raised ValidationError unexpectedly: {e}")

    assert simulation_day == 0


def test_simulation_date_without_dynode_flag():
    with pytest.raises(ValueError):
        config.simulation_day(2022, 2, 11)


def test_simulation_date_with_dynode_flag():
    config.set_dynode_init_date_flag(date(2022, 2, 11))
    simulation_date = config.simulation_day(2022, 2, 11)
    assert simulation_date == 0, (
        "Sim day should be 0 when dynode init date flag is set to same day"
    )


def test_simulation_date_with_future_date():
    config.set_dynode_init_date_flag(date(2022, 2, 1))
    simulation_date = config.simulation_day(2022, 2, 11)
    assert simulation_date == 10, (
        "Sim day should be 0 when dynode init date flag is set to same day"
    )


def test_replace_simulation_dates_dict():
    config.set_dynode_init_date_flag(date(2022, 2, 11))
    obj = {
        "simulation_date": config.simulation_day(2022, 2, 11),
        "other_field": "test_value",
    }
    assert obj["simulation_date"] == 0, (
        "replace_simuoation_dates is not finding SimulationDate objects correctly."
    )


def test_replace_simulation_dates_obj():
    # make a test pydantic class to search
    config.set_dynode_init_date_flag(date(2022, 2, 11))

    class test(BaseModel):
        simulation_day: int

    config.set_dynode_init_date_flag(date(2022, 2, 11))
    obj = test(simulation_day=config.simulation_day(2022, 2, 11))
    assert obj.simulation_day == 0, (
        "replace_simuoation_dates is not finding SimulationDate objects correctly."
    )


def test_replace_simulation_dates_list():
    config.set_dynode_init_date_flag(date(2022, 2, 11))
    obj = [config.simulation_day(2022, 2, 11), "test_value"]
    assert obj[0] == 0, (
        "replace_simuoation_dates is not finding SimulationDate objects correctly."
    )


def test_replace_simulation_dates_distribution():
    config.set_dynode_init_date_flag(date(2022, 2, 11))
    obj = Normal(loc=config.simulation_day(2022, 2, 11), scale=1.0)
    assert obj.loc == 0, (
        "replace_simulation_dates is not finding SimulationDate objects correctly in distributions."
    )
    try:
        obj.sample(key=PRNGKey(0))
    except Exception as e:
        pytest.fail(
            f"Sampling from distribution with SimulationDate failed: {e}"
        )


def test_replace_simulation_dates_custom_object():
    class custom_obj:
        def __init__(self):
            self.x: int = config.simulation_day(2022, 2, 11)

    config.set_dynode_init_date_flag(date(2022, 2, 11))
    custom_o = custom_obj()
    assert custom_o.x == 0, (
        "failed to replace simulation dates in a custom object"
    )


def test_replace_simulation_dates_nested_distributions():
    config.set_dynode_init_date_flag(date(2022, 2, 11))
    obj = TruncatedNormal(
        loc=config.simulation_day(2022, 2, 11),
        scale=1.0,
        low=config.simulation_day(2022, 2, 10),
        high=config.simulation_day(2022, 2, 12),
    )

    assert obj.base_dist.loc == 0, (
        "replace_simulation_dates is not finding SimulationDate objects correctly in nested distributions."
    )
    assert obj.low == -1
    assert obj.high == 1

    try:
        obj.sample(key=PRNGKey(0))
    except Exception as e:
        pytest.fail(
            f"Sampling from distribution with SimulationDate failed: {e}"
        )


def test_replace_simulation_dates_skips_frozen_dataclasses():
    config.set_dynode_init_date_flag(date(2022, 2, 11))

    @dataclass(frozen=True)
    class custom_obj:
        x: int

    custom_o = custom_obj(config.simulation_day(2022, 2, 11))
    assert isinstance(custom_o.x, int)
    assert custom_o.x == 0
