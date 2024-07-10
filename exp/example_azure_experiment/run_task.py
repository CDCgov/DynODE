"""
An example script similar to example_end_to_end_run.py
but adapted to show the differences between Azure runs and local ones.
"""

# ruff: noqa: E402
import argparse
import sys

sys.path.append("/app/")

from mechanistic_model.abstract_azure_runner import AbstractAzureRunner
from mechanistic_model.covid_sero_initializer import CovidSeroInitializer
from mechanistic_model.mechanistic_runner import MechanisticRunner
from mechanistic_model.static_value_parameters import StaticValueParameters
from model_odes.seip_model import seip_ode


class ExampleRunner(AbstractAzureRunner):
    # __init__ already implemented by the abstract case
    def __init__(self, azure_output_dir):
        super().__init__(azure_output_dir)

    # override the process state command since this is what
    # each individual runner must figure out for itself
    def process_state(self, state):
        """
        A similar version to example_end_to_end_run.py but modified to run on Azure batch

        Parameters
        ---------------
        state: str
            the USPS postal code of the state, e.g. CA, OR, NY, TX
        """
        # step 1: define your paths NOTE: These are all within the docker container!
        # /input is a MOUNTED drive that we upload these files into right before the job launched
        config_path = "/input/exp/example_azure_experiment/states/%s/" % state
        # global_config include definitions such as age bin bounds and strain definitions
        # Any value or data structure that needs context to be interpretted is here.
        GLOBAL_CONFIG_PATH = config_path + "config_global.json"
        # defines the init conditions of the scenario: pop size, initial infections etc.
        INITIALIZER_CONFIG_PATH = config_path + "config_initializer_covid.json"
        # defines the running variables, strain R0s, external strain introductions etc.
        RUNNER_CONFIG_PATH = config_path + "config_runner_covid.json"
        self.save_config(GLOBAL_CONFIG_PATH)
        self.save_config(INITIALIZER_CONFIG_PATH)
        self.save_config(RUNNER_CONFIG_PATH)
        # sets up the initial conditions, initializer.get_initial_state() passed to runner
        initializer = CovidSeroInitializer(
            INITIALIZER_CONFIG_PATH, GLOBAL_CONFIG_PATH
        )
        # reads and interprets values from config, sets up downstream parameters
        # like beta = STRAIN_R0s / INFECTIOUS_PERIOD
        static_params = StaticValueParameters(
            initializer.get_initial_state(),
            RUNNER_CONFIG_PATH,
            GLOBAL_CONFIG_PATH,
        )
        # A runner that does ODE solving of a single run.
        runner = MechanisticRunner(seip_ode)
        # run for 200 days, using init state and parameters from StaticValueParameters
        solution = runner.run(
            initializer.get_initial_state(),
            tf=200,
            args=static_params.get_parameters(),
        )
        self.save_static_run_timelines(static_params, solution)
        return solution


parser = argparse.ArgumentParser()
parser.add_argument(
    "-s",
    "--state",
    type=str,
    help="directory for the state to run, resembles USPS code of the state",
    required=True,
)

parser.add_argument(
    "-j",
    "--jobid",
    type=str,
    help="job-id of the state being run on Azure",
    required=True,
)
args = parser.parse_args()
state = args.state
jobid = args.jobid
save_path = "/output/example_output/%s/%s/" % (jobid, state)
runner = ExampleRunner(save_path)
runner.process_state(state)
