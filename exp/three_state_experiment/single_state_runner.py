# ruff: noqa: E402
import argparse
import os
import sys

sys.path.append("/app")
sys.path.append("/app/mechanistic_model/")
print(sys.path)
import matplotlib.pyplot as plt

from mechanistic_model.covid_initializer import CovidInitializer
from mechanistic_model.mechanistic_runner import MechanisticRunner
from mechanistic_model.solution_iterpreter import SolutionInterpreter
from mechanistic_model.static_value_parameters import StaticValueParameters
from model_odes.seip_model import seip_ode

parser = argparse.ArgumentParser()
parser.add_argument(
    "-s",
    "--state",
    type=str,
    help="directory for the state to run, resembles USPS code of the state",
)

if __name__ == "__main__":
    args = parser.parse_args()
    # step 1: define your paths
    state_config_path = "/app/exp/three_state_experiment/" + args.state + "/"
    print("Running the following state: " + str(args.state) + "\n")
    # global_config include definitions such as age bin bounds and strain definitions
    # Any value or data structure that needs context to be interpretted is here.
    GLOBAL_CONFIG_PATH = state_config_path + "config_global.json"
    # defines the init conditions of the scenario: pop size, initial infections etc.
    INITIALIZER_CONFIG_PATH = (
        state_config_path + "config_initializer_covid.json"
    )
    # defines the running variables, strain R0s, external strain introductions etc.
    RUNNER_CONFIG_PATH = state_config_path + "config_runner_covid.json"
    # defines prior __distributions__ for inferring runner variables.
    INFERER_CONFIG_PATH = state_config_path + "config_inferer_covid.json"
    # defines how the solution should be viewed, what slices examined, how to save.
    INTERPRETER_CONFIG_PATH = (
        state_config_path + "config_interpreter_covid.json"
    )
    # sets up the initial conditions, initializer.get_initial_state() passed to runner
    initializer = CovidInitializer(INITIALIZER_CONFIG_PATH, GLOBAL_CONFIG_PATH)
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

    # interpret the solution object in a variety of ways
    interpreter = SolutionInterpreter(
        solution, INTERPRETER_CONFIG_PATH, GLOBAL_CONFIG_PATH
    )
    # plot the 4 compartments summed across all age bins and immunity status
    fig, ax = interpreter.summarize_solution()
    save_path = (
        "/output/three_state_experiment/%s/example_end_to_end_run_.png"
        % args.state
    )
    if not os.path.exists(save_path):
        os.makedirs("/output/three_state_experiment/%s" % args.state)
    print("Please see %s for your plot!" % save_path)
    plt.savefig(save_path)
