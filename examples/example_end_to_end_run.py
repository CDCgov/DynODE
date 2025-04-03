#!/usr/bin/env python
"""A basic covid example to show off how to work with DynODE.

Results produced by this basic example are not meant to be taken as serious
predictions, but rather a demonstration with synthetic data.

To see a image detailing the output of a single run, if you set the --infer flag
this script will generate some example output, and then fit back onto it with some broad prior
estimates of what epidemiological variables produced it.

For runtime the fitting is very short and not very accurate, you may improve
the accuracy at the cost of runtime by modifying the inferer config.
"""

import argparse
import os

import jax.numpy as jnp
import numpy as np

# the different segments of code responsible for runing the model
# each will be explained as they are used below
from dynode import (  # type: ignore
    AbstractDynodeRunner,
    CovidSeroInitializer,
    MechanisticInferer,
    MechanisticRunner,
    StaticValueParameters,
    vis_utils,
)
from dynode.model_odes import seip_ode  # type: ignore
from dynode.utility import log

parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--infer",
    default=False,
    action="store_true",
    help="whether or not to run the inference section of this example",
)

subparsers = parser.add_subparsers(title="subcommands", dest="subcommand")

log_parser = subparsers.add_parser("log", help="Subcommands for logging")
log_parser.add_argument(
    "-l",
    "--level",
    default="info",
    choices=["debug", "info", "warning", "error", "critical"],
    help="set the logging level the default if info",
)
log_parser.add_argument(
    "-o",
    "--output",
    default="file",
    choices=["file", "console", "both"],
    help="print logs to console, file, or both the default is file",
)


class ExampleDynodeRunner(AbstractDynodeRunner):
    """An Example DynODE Runner used to launch this illustration."""

    def process_state(self, state: str, **kwargs):
        """Every DynODE runner must have an entry point.

        In this case the example configs are built around US data, so we are
        running the whole US as a single entity.

        Parameters
        ----------
        state : str
            State USPS, ignored in this specific example as it only runs USA.
        infer : bool, optional
            Whether or not the user of this example script wants to run inference,
            by default False. Stored in kwargs to align with
            `AbstractDynodeRunner` function signature.
        """
        infer = bool(kwargs["infer"])
        # step 1: define your paths
        config_path = "examples/config/"
        # global_config include definitions such as age bin bounds and strain definitions
        # Any value or data structure that needs context to be interpretted is here.
        GLOBAL_CONFIG_PATH = config_path + "config_global.json"
        # defines the init conditions of the scenario: pop size, initial infections etc.
        INITIALIZER_CONFIG_PATH = config_path + "config_initializer_covid.json"
        # defines the running variables, strain R0s, external strain introductions etc.
        RUNNER_CONFIG_PATH = config_path + "config_runner_covid.json"
        # defines prior distributions for inferring variables.
        INFERER_CONFIG_PATH = config_path + "config_inferer_covid.json"

        # step 2: Set up your initializer
        # sets up the initial conditions, initializer.get_initial_state() passed to runner
        initializer = CovidSeroInitializer(
            INITIALIZER_CONFIG_PATH, GLOBAL_CONFIG_PATH
        )

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
            assert (
                solution.ys is not None
            ), "solution.ys returned None, odes failed."
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
            inferer.set_infer_algo()
            print("Fitting to synthetic hospitalization data: ")
            # this will print a summary of the inferred variables
            # those distributions in the Config are now posteriors
            inferer.infer(jnp.array(synthetic_observed_hospitalizations))
            print("saving a suite of inference visualizations ")
            # save some particle 0 and 5 from chains 0 and 1 for example
            self.generate_inference_timeseries(
                inferer,
                particles=[(0, 0), (1, 0), (0, 5), (1, 5)],
                timeseries_filename="local_inference_timeseries.csv",
            )
            self.save_inference_posteriors(
                inferer, "local_example_inferer_posteriors.json"
            )
            print(
                "Toy inference finished, see the distributions of posteriors above, "
                "in only 60 samples how well do they match with the actual parameters "
                "used to generate the synthetic data? Likely not well..., try "
                "to increase the INFERENCE_NUM_SAMPLES and INFERENCE_NUM_WARMUP "
                "parameters in the config_inferer_covid.json to see this improve. \n"
            )
            print(
                "static values used to generate synthetic hosp data: \n"
                "INFECTIOUS_PERIOD : %s \n"
                "INTRODUCTION_TIMES_BA2/5 : %s \n"
                "IHRS : %s"
                % (
                    static_params.config.INFECTIOUS_PERIOD,
                    static_params.config.INTRODUCTION_TIMES[0],
                    str(ihr),
                )
            )
        else:
            # step 5: interpret the solution object in a variety of ways
            save_path = "output/example_end_to_end_run.png"
            df = self.save_static_run_timeseries(
                static_params, solution, "local_run_timeseries.csv"
            )
            # attach a `state` column so plot cols have titles
            df["state"] = "USA"
            # for normalization of metrics per 100k
            usa_pop = {"USA": initializer.config.POP_SIZE}
            fig = vis_utils.plot_model_overview_subplot_matplotlib(df, usa_pop)
            print("Please see %s for your plot!" % save_path)
            fig.savefig(save_path)


if __name__ == "__main__":
    args = parser.parse_args()
    infer = args.infer

    # Make the output directory.
    if not os.path.exists("output"):
        os.mkdir("output")

    if args.subcommand == "log":
        log.use_logging(level=args.level, output=args.output)

    runner = ExampleDynodeRunner("output/")
    runner.process_state("USA", infer=infer)
