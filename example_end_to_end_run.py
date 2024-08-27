"""
A basic local covid example meant to show off the flow of running the mechanistic model

Results produced by this basic example are not meant to be taken as serious predictions, but rather
a demonstration with synthetic data.

To see a image detailing the output of a single run, set the --infer flag to False, otherwise
this script will generate some example output, and then fit back onto it with some broad prior
estimates of what epidemiological variables produced it.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

# the different segments of code responsible for runing the model
# each will be explained as they are used below
from mechanistic_model.covid_initializer import CovidInitializer
from mechanistic_model.mechanistic_inferer import MechanisticInferer
from mechanistic_model.mechanistic_runner import MechanisticRunner
from mechanistic_model.solution_iterpreter import SolutionInterpreter
from mechanistic_model.static_value_parameters import StaticValueParameters
from model_odes.seip_model import seip_ode

parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--infer",
    type=bool,
    help="Whether or not to run the the inference section of this example",
)

if __name__ == "__main__":
    args = parser.parse_args()
    infer = args.infer

    # step 1: define your paths
    config_path = "config/"
    # global_config include definitions such as age bin bounds and strain definitions
    # Any value or data structure that needs context to be interpretted is here.
    GLOBAL_CONFIG_PATH = config_path + "config_global.json"
    # defines the init conditions of the scenario: pop size, initial infections etc.
    INITIALIZER_CONFIG_PATH = config_path + "config_initializer_covid.json"
    # defines the running variables, strain R0s, external strain introductions etc.
    RUNNER_CONFIG_PATH = config_path + "config_runner_covid.json"
    # defines prior distributions for inferring variables.
    INFERER_CONFIG_PATH = config_path + "config_inferer_covid.json"
    # defines how the solution should be viewed, what slices examined, how to save.
    INTERPRETER_CONFIG_PATH = config_path + "config_interpreter_covid.json"

    # step 2: Set up your initializer
    # sets up the initial conditions, initializer.get_initial_state() passed to runner
    initializer = CovidInitializer(INITIALIZER_CONFIG_PATH, GLOBAL_CONFIG_PATH)

    # step 3: set up your parameters object
    # reads and interprets values from config, sets up downstream parameters
    # like beta = STRAIN_R0s / INFECTIOUS_PERIOD
    static_params = StaticValueParameters(
        initializer.get_initial_state(),
        RUNNER_CONFIG_PATH,
        GLOBAL_CONFIG_PATH,
    )

    # step 4: set up your runner and solve ODE equation
    # A runner that does ODE solving of a single run.
    runner = MechanisticRunner(seip_ode)
    # run for 200 days, using init state and parameters from StaticValueParameters
    solution = runner.run(
        initializer.get_initial_state(),
        tf=200,
        args=static_params.get_parameters(),
    )
    if infer:
        # for an example inference, lets jumble our solution up a bit and attempt to fit back to it
        # apply an age specific IHR to the infection incidence to get hospitalization incidence
        ihr = [0.002, 0.004, 0.008, 0.06]
        model_incidence = np.sum(solution.ys[3], axis=(2, 3, 4))
        model_incidence = np.diff(model_incidence, axis=0)
        # noise our "hospitalization" data with a negative binomial distribution
        rng = np.random.default_rng(seed=8675309)
        m = np.asarray(model_incidence) * ihr
        k = 10.0
        p = k / (k + m)
        synthetic_observed_hospitalizations = rng.negative_binomial(k, p)

        # step 4B: set up yout inferer, defining prior distributions of some parameters
        inferer = MechanisticInferer(
            GLOBAL_CONFIG_PATH,
            INFERER_CONFIG_PATH,
            runner,
            initializer.get_initial_state(),
        )
        # artificially shortening inference since this is a toy example
        inferer.config.INFERENCE_NUM_WARMUP = 30
        inferer.config.INFERENCE_NUM_SAMPLES = 30
        inferer.set_infer_algo()
        print("Fitting to synthetic hospitalization data: ")
        # this will print a summary of the inferred variables
        # those distributions in the Config are now posteriors
        inferer.infer(synthetic_observed_hospitalizations)
        print(
            "Toy inference finished, see the distributions of posteriors above, "
            "in only 60 samples how well do they match with the actual parameters "
            "used to generate the fake data? \n"
        )
    else:
        # step 5: interpret the solution object in a variety of ways
        interpreter = SolutionInterpreter(
            solution, INTERPRETER_CONFIG_PATH, GLOBAL_CONFIG_PATH
        )
        # plot the 4 compartments summed across all age bins and immunity status
        fig, ax = interpreter.summarize_solution()
        save_path = "output/example_end_to_end_run.png"
        print("Please see %s for your plot!" % save_path)
        plt.savefig(save_path)
