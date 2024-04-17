"""
Integration tests are done here.

This file is responsible for creating copies of each config JSON file within COPIED_TEMP_FILES,
passing those copies to each test, and removing them from memory after the test concludes.

To follow integration test best practices, avoid modifying model parameters
in the models themselves, instead modify config parameters in the JSON where at all possible.
"""

import datetime
import json
import os
import tempfile

import jax.numpy as jnp
import numpy as np
import pytest

from mechanistic_model.covid_initializer import CovidInitializer
from mechanistic_model.mechanistic_runner import MechanisticRunner
from mechanistic_model.static_value_parameters import StaticValueParameters
from model_odes.seip_model import seip_ode

CONFIG_GLOBAL_PATH = "tests/test_config_global.json"
INITIALIZER_CONFIG_PATH = "tests/test_config_initializer.json"
RUNNER_CONFIG_PATH = "tests/test_config_runner.json"
INFERER_CONFIG_PATH = "tests/test_config_inferer.json"
INTERPRETER_CONFIG_PATH = "tests/test_config_interpreter.json"

CONFIG_GLOBAL = json.loads(open(CONFIG_GLOBAL_PATH, "r").read())
CONFIG_INITIALIZER = json.loads(open(INITIALIZER_CONFIG_PATH, "r").read())
CONFIG_INFERER = json.loads(open(INFERER_CONFIG_PATH, "r").read())
CONFIG_INTERPRETER = json.loads(open(INTERPRETER_CONFIG_PATH, "r").read())
CONFIG_RUNNER = json.loads(open(RUNNER_CONFIG_PATH, "r").read())

COPIED_TEMP_FILES = [
    CONFIG_GLOBAL,
    CONFIG_INITIALIZER,
    CONFIG_INFERER,
    CONFIG_INTERPRETER,
    CONFIG_RUNNER,
]


@pytest.fixture(scope="function")
def temp_config_files(request):
    """
    will run before each test starts, creating temp copies of each config json files defined at the top of this file.
    These temp file objects will be created and deleted during a single test, and
    """
    copied_file_paths = []
    for coped_temp_file in COPIED_TEMP_FILES:
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as t:
            copied_file_paths.append(t.name)
            json.dump(coped_temp_file, t)

    def cleanup():
        # Clean up the temporary file after the test is done with it
        for copied_file_path in copied_file_paths:
            os.remove(copied_file_path)

    request.addfinalizer(cleanup)
    return tuple(copied_file_paths)


def test_temp_config_functionality(temp_config_files):
    # make sure the order matches the order of COPIED_TEMP_FILES
    (
        temp_global_path,
        temp_initializer_path,
        temp_inferer_path,
        temp_interpreter_path,
        temp_runner_path,
    ) = temp_config_files
    # load in the temp global file
    global_config = json.load(open(temp_global_path, "r"))
    # change something about it and push those changes back
    global_config["MINIMUM_AGE"] = 25
    json.dump(global_config, open(temp_global_path, "w"))
    # reload the file and make sure the changes are now there
    global_config = json.load(open(temp_global_path, "r"))
    assert global_config["MINIMUM_AGE"] == 25, "changes not pushed"


def test_vaccination_rates(temp_config_files):
    """
    creates a scenario where entire population is 1 person and removes all infected and exposed people
    turns off all infections and external introductions of infected people.

    Then runs the odes for one timestep and asserts that the vaccination splines are followed correctly.

    Population is set to 1 because all vaccination rates are normalized to the total population size (1).
    """
    # make sure the order matches the order of COPIED_TEMP_FILES
    (
        temp_global_path,
        temp_initializer_path,
        _,
        _,
        temp_runner_path,
    ) = temp_config_files
    # creating the scenario
    initializer = json.load(open(temp_initializer_path, "r"))
    initializer["POP_SIZE"] = 1  # pop size to 1
    initializer["INITIAL_INFECTIONS"] = 0  # no initial infections
    # saving initializer changes
    json.dump(initializer, open(temp_initializer_path, "w"))
    runner = json.load(open(temp_runner_path, "r"))
    runner["STRAIN_R0s"] = [0.0, 0.0, 0.0]  # no new infections
    runner["INTRODUCTION_TIMES"] = []  # turn off external introductions
    runner["INTRODUCTION_SCALES"] = []  # turn off external introductions
    runner["INTRODUCTION_PERCS"] = []  # turn off external introductions
    # saving runner changes
    json.dump(runner, open(temp_runner_path, "w"))

    # integration test
    initializer = CovidInitializer(temp_initializer_path, temp_global_path)
    static_params = StaticValueParameters(
        initializer.get_initial_state(), temp_runner_path, temp_global_path
    )
    # test vaccination vs ode for 10 timesteps
    # we dont use the MechanisticRunner here because the adaptive step size can mess with things
    next_state = static_params.INITIAL_STATE
    for t in range(10):
        cur_state = next_state
        compartment_changes = seip_ode(
            cur_state, t, static_params.get_parameters()
        )
        # get s compartment, since rest are zeros anyways
        # normalize by proportion of each age bin of total population
        ds = (
            np.sum(compartment_changes[0], axis=(1, 3))
            / initializer.config.INITIAL_POPULATION_FRACTIONS[:, None]
        )
        vaccination_rates = static_params.vaccination_rate(t)
        # max vax always just goes back into itself, we set this to zero
        # as the net movement from max -> max is zero
        vaccination_rates = vaccination_rates.at[:, -1].set(0)
        for dose in range(static_params.config.MAX_VAX_COUNT + 1):
            if dose == 0:
                assert (
                    np.isclose(ds[:, dose], -vaccination_rates[:, dose])
                ).all(), (
                    "Vaccination flows incorrect when vaccinating 0-> 1 dose Vaccination Rates Received: %s \n changes to suseptible populations compared %s. keep in mind vaccination_rate(t)[:, -1] is zeroed out since that dose flows into itself"
                    % (
                        str(vaccination_rates),
                        str(ds),
                    )
                )
            else:
                assert (
                    np.isclose(
                        ds[:, dose],
                        vaccination_rates[:, dose - 1]
                        - vaccination_rates[:, dose],
                    )
                ).all(), (
                    "vaccination flows incorrect when vaccinating x-> x+1 dose x> 0. Vaccination Rates Received: %s \n changes to suseptible populations compared %s keep in mind vaccination_rate(t)[:, -1] is zeroed out since that dose flows into itself"
                    % (
                        str(vaccination_rates),
                        str(ds),
                    )
                )
        # propagate the changes to the next state for the ODE to run again
        next_state = tuple(
            [
                cur_state[i] + compartment_changes[i]
                for i in static_params.config.COMPARTMENT_IDX
            ]
        )


def test_seasonal_vaccination(temp_config_files):
    """
    creates a scenario where entire population is 1 person and removes all infected and exposed people
    turns off all infections, external introductions of infected people, and new vaccination.

    checks to make sure number of seasonal-vaccinated individuals after the
    vaccination season ends is less than 0.1% of the total population.

    This test assumes some level of initial vaccination in the model
    """
    # make sure the order matches the order of COPIED_TEMP_FILES
    (
        temp_global_path,
        temp_initializer_path,
        _,
        _,
        temp_runner_path,
    ) = temp_config_files
    # creating the scenario
    global_config = json.load(open(temp_global_path, "r"))
    # we want to initialize the season and then immediately end vaccine season
    global_config["VAX_SEASON_CHANGE"] = (
        datetime.datetime.strptime(global_config["INIT_DATE"], "%Y-%m-%d")
        + datetime.timedelta(days=15)
    ).strftime("%Y-%m-%d")
    # saving global settings
    json.dump(global_config, open(temp_global_path, "w"))
    initializer = json.load(open(temp_initializer_path, "r"))
    initializer["POP_SIZE"] = 1  # pop size to 1
    initializer["INITIAL_INFECTIONS"] = 0  # no initial infections
    # saving initializer changes
    json.dump(initializer, open(temp_initializer_path, "w"))
    runner = json.load(open(temp_runner_path, "r"))
    runner["STRAIN_R0s"] = [0.0, 0.0, 0.0]  # no new infections
    runner["INTRODUCTION_TIMES"] = []  # turn off external introductions
    runner["INTRODUCTION_SCALES"] = []  # turn off external introductions
    runner["INTRODUCTION_PERCS"] = []  # turn off external introductions
    runner["SEASONAL_VACCINATION"] = True  # turn on seasonal vaccination
    # saving runner changes
    json.dump(runner, open(temp_runner_path, "w"))

    # integration test
    initializer = CovidInitializer(temp_initializer_path, temp_global_path)
    static_params = StaticValueParameters(
        initializer.get_initial_state(), temp_runner_path, temp_global_path
    )
    # overriding the coefficients to be all zeros, effectively turning off vaccination
    static_params.config.VAX_MODEL_KNOTS = jnp.zeros(
        (
            static_params.config.NUM_AGE_GROUPS,
            static_params.config.MAX_VAX_COUNT + 1,
            static_params.config.VAX_MODEL_NUM_KNOTS,
        )
    )
    runner = MechanisticRunner(seip_ode)
    # run for 50 days and witness the season end at t=5
    solution = runner.run(
        initial_state=initializer.get_initial_state(),
        args=static_params.get_parameters(),
        tf=50,
    )
    s_compartment = solution.ys[static_params.config.COMPARTMENT_IDX.S]
    num_seasonal_vax_at_t_0 = np.sum(
        s_compartment[0, :, :, static_params.config.MAX_VAX_COUNT, :]
    )
    # almost all of the individuals should be moved out of the seasonal vax tier by t=6
    # we allow some tiny number of people who got vaccinated for the next season on that tier
    # to exist there, but it should be tiny.
    num_seasonal_vax_at_t_20 = np.sum(
        s_compartment[20, :, :, static_params.config.MAX_VAX_COUNT, :]
    )
    error_msg = """The day after the vaccine season ends, you still have a measurable
    number of individuals left in the seasonal vaccination tier. Had %s individuals in seasonal tier initially
    vs %s individuals in the seasonal vax tier after the end of the season. This value should be near zero.""" % (
        num_seasonal_vax_at_t_0,
        num_seasonal_vax_at_t_20,
    )
    assert num_seasonal_vax_at_t_20 < 0.001, error_msg
