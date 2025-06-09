import os
from datetime import date

import pytest
from jax.random import PRNGKey
from numpyro.distributions import Normal
from pydantic import BaseModel, ConfigDict, ValidationError

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
        simulation_date = config.SimulationDate(2022, 2, 11)
    except ValidationError as e:
        pytest.fail(f"SimulationDate raised ValidationError unexpectedly: {e}")

    assert simulation_date.year == 2022


def test_simulation_date_without_dynode_flag():
    simulation_date = config.SimulationDate(2022, 2, 11)
    with pytest.raises(ValueError):
        simulation_date.sim_day


def test_simulation_date_with_dynode_flag():
    simulation_date = config.SimulationDate(2022, 2, 11)
    config.simulation_date.set_dynode_init_date_flag(date(2022, 2, 11))
    sim_day = simulation_date.sim_day
    assert (
        sim_day == 0
    ), "Sim day should be 0 when dynode init date flag is set to same day"


def test_simulation_date_with_future_date():
    simulation_date = config.SimulationDate(2022, 2, 11)
    config.simulation_date.set_dynode_init_date_flag(date(2022, 2, 1))
    sim_day = simulation_date.sim_day
    assert (
        sim_day == 10
    ), "Sim day should be 0 when dynode init date flag is set to same day"


def test_replace_simulation_dates_dict():
    obj = {
        "simulation_date": config.SimulationDate(2022, 2, 11),
        "other_field": "test_value",
    }
    config.simulation_date.set_dynode_init_date_flag(date(2022, 2, 11))
    obj = config.simulation_date.replace_simulation_dates(obj)
    assert (
        obj["simulation_date"] == 0
    ), "replace_simuoation_dates is not finding SimulationDate objects correctly."


def test_replace_simulation_dates_obj():
    # make a test pydantic class to search
    class test(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)
        simulation_date: config.SimulationDate

    obj = test(simulation_date=config.SimulationDate(2022, 2, 11))
    config.simulation_date.set_dynode_init_date_flag(date(2022, 2, 11))
    obj = config.simulation_date.replace_simulation_dates(obj)
    assert (
        obj.simulation_date == 0
    ), "replace_simuoation_dates is not finding SimulationDate objects correctly."


def test_replace_simulation_dates_list():
    obj = [config.SimulationDate(2022, 2, 11), "test_value"]
    config.simulation_date.set_dynode_init_date_flag(date(2022, 2, 11))
    obj = config.simulation_date.replace_simulation_dates(obj)
    assert (
        obj[0] == 0
    ), "replace_simuoation_dates is not finding SimulationDate objects correctly."


def test_replace_simulation_dates_distribution():
    obj = Normal(loc=config.SimulationDate(2022, 2, 11), scale=1.0)
    config.simulation_date.set_dynode_init_date_flag(date(2022, 2, 11))
    obj: Normal = config.simulation_date.replace_simulation_dates(obj)
    assert (
        obj.loc == 0
    ), "replace_simulation_dates is not finding SimulationDate objects correctly in distributions."

    try:
        obj.sample(key=PRNGKey(0))
    except Exception as e:
        pytest.fail(
            f"Sampling from distribution with SimulationDate failed: {e}"
        )
