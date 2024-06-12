"""
The following abstract class defines a mechanistic_initializer,
which is used by mechanistic_runner instances to initialize the state on which ODEs will be run.
mechanistic_initializers will often be tasked with reading, parsing, and combining data sources
to produce an initial state representing some analyzed population
"""

import json
import os
from abc import ABC, abstractmethod

import jax.numpy as jnp
import numpy as np
from jax import Array
from mechanistic_inferer import MechanisticInferer

import utils


class AbstractAzureRunner(ABC):
    def __init__(self, azure_output_dir):
        # saving for future use
        self.azure_output_dir = azure_output_dir
        if not os.path.exists(azure_output_dir):
            os.makedirs(azure_output_dir, exist_ok=True)
        # create two dual loggers to save sys.stderr and sys.stdout to files in `azure_output_dir`
        utils.dual_logger_out(
            os.path.join(azure_output_dir, "stdout.txt"), "w"
        )
        utils.dual_logger_err(
            os.path.join(azure_output_dir, "stderr.txt"), "w"
        )

    @abstractmethod
    def process_state(self):
        """
        Abstract function meant to be implemented by the instance of the runner.
        This handles all of the logic of actually getting a solution object.
        """
        pass

    def save_inference_visuals(
        self,
        inferer: MechanisticInferer,
        vis_filename: str = "azure_visualizer_inference_timeline.json",
    ) -> str:
        """
        saves history of inferer sampled values for use by the azure visualizer.
        saves JSON file to `self.azure_output_dir/vis_path`.
        Look at shiny_visualizers/azure_visualizer.py for logic on parsing and visualizing the chains.

        Parameters
        ------------
        inferer: MechanisticInferer
            the inferer object used to sample the parameter chains that will be visualized

        Returns
        ------------
        a str path of where the json file was saved to.
        """
        # TODO this method currently does not modify the parameter names using the enums found in `inferer`
        # eg. STRAIN_R0s_0 should be represented as STRAIN_R0s_strain
        # where strain is found in inferer.config.STRAIN_IDX._member_names_[0]
        # this is not currently implemented as it requires some mapping of parameter indexes to names
        inference_visuals_save_path = os.path.join(
            self.azure_output_dir,
            vis_filename,
        )
        # dictionary of sampled parameters str: np.array, save the chains
        sampled_parameters = inferer.inference_algo._states[
            inferer.inference_algo._sample_field
        ]
        # flatten the parameters so no numpyro.plate() parameters get through
        sampled_parameters = utils.flatten_list_parameters(sampled_parameters)
        # convert all the np and jnp arrays to list so Jax can parse them
        for parameter in sampled_parameters.keys():
            if isinstance(
                sampled_parameters[parameter], (np.ndarray, jnp.ndarray)
            ):
                sampled_parameters[parameter] = sampled_parameters[
                    parameter
                ].tolist()
        # dump and return the path
        with open(inference_visuals_save_path, "w") as file:
            json.dump(sampled_parameters, file)

        return inference_visuals_save_path

    def save_single_run_visuals(
        self,
        sol: tuple[Array, Array, Array, Array],
        vis_filename: str = "azure_visualizer_inference_timeline.json",
    ):
        """
        given a tuple of compartment timelines, saves a number of timelines of interest for future visualization
        usually `sol` is retrieved from `diffrax.Solution.ys` which is an object returned from `MechanisticRunner.run()`
        """
        inference_visuals_save_path = os.path.join(
            self.azure_output_dir,
            vis_filename,
        )
        return inference_visuals_save_path

    def save_model_component_timelines():
        """
        given an parameters object inheriting from `AbstractParameters` plots a number of different timelines
        """
