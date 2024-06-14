"""
The following abstract class defines a mechanistic_initializer,
which is used by mechanistic_runner instances to initialize the state on which ODEs will be run.
mechanistic_initializers will often be tasked with reading, parsing, and combining data sources
to produce an initial state representing some analyzed population
"""

import os
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pandas as pd
from jax import Array

import utils
from mechanistic_model.abstract_parameters import AbstractParameters
from mechanistic_model.mechanistic_inferer import MechanisticInferer
from mechanistic_model.solution_iterpreter import SolutionInterpreter


class AbstractAzureRunner(ABC):
    def __init__(self, azure_output_dir):
        # saving for future use
        self.azure_output_dir = azure_output_dir
        if not os.path.exists(azure_output_dir):
            os.makedirs(azure_output_dir, exist_ok=True)
        # create two dual loggers to save sys.stderr and sys.stdout to files in `azure_output_dir`
        utils.dual_logger_out(os.path.join(azure_output_dir, "stdout.txt"), "w")
        utils.dual_logger_err(os.path.join(azure_output_dir, "stderr.txt"), "w")

    @abstractmethod
    def process_state(self, state):
        """
        Abstract function meant to be implemented by the instance of the runner.
        This handles all of the logic of actually getting a solution object.
        """
        pass

    def save_inference_visuals(
        self,
        inferer: MechanisticInferer,
        solution_interpreter: SolutionInterpreter,
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
            a str path of where the json file was saved to. Or RuntimeError if inference has not been
            completed by the `inferer` yet.
        """
        # TODO this method currently does not modify the parameter names using the enums found in `inferer`
        # eg. STRAIN_R0s_0 should be represented as STRAIN_R0s_strain
        # where strain is found in inferer.config.STRAIN_IDX._member_names_[0]
        # this is not currently implemented as it requires some mapping of parameter indexes to names
        inference_visuals_save_path = os.path.join(
            self.azure_output_dir,
            vis_filename,
        )
        # assuming the inferer has finished fitting
        posteriors = inferer.load_posterior_particle(0)
        # for (particle, chain), sol_dct in posteriors.items():
        #     infection_timeline, posterior_values = (
        #         sol_dct["solution"],
        #         sol_dct["posteriors"],
        #     )
        #     hospitalizations = 1

        return inference_visuals_save_path

    def save_single_run_visuals(
        self,
        sol: tuple[Array, Array, Array, Array],
        solution_interpreter: SolutionInterpreter,
        vis_filename: str = "azure_visualizer_inference_timeline.png",
    ):
        """
        given a tuple of compartment timelines, saves a number of timelines of interest for future visualization
        usually `sol` is retrieved from `diffrax.Solution.ys` which is an object returned from `MechanisticRunner.run()`
        """
        inference_visuals_save_path = os.path.join(
            self.azure_output_dir,
            vis_filename,
        )
        # plot the 4 compartments summed across all age bins and immunity status
        _ = solution_interpreter.summarize_solution()
        return inference_visuals_save_path

    def _generate_model_component_timelines(
        self,
        parameters: AbstractParameters,
        infections: tuple[Array, Array, Array, Array],
        hospitalization_preds: tuple[Array, Array, Array, Array],
        hospitalization_ground_truth: Union[np.ndarray, None] = None,
    ) -> pd.DataFrame:
        """
        a private function which takes two timelines of infections and hospitalizations and generates a
        dataframe of different timelines of interest

        Parameters
        ----------
        parameters : AbstractParameters
            a class that inherits from AbstractParameters and therefore has a get_parameters() method
        infections : tuple[Array, Array, Array, Array]
            compartments sizes of the model as produced by MechanisticRunner.run() commands
        hospitalization_preds : tuple[Array, Array, Array, Array]
            models hospitalization predictions, usually the infections
            matrix with some infection hospitalization ratio applied
        hospitalization_ground_truth: Optional np.ndarray
            Optional observed hospitalization by age group,
            timeline allowed to be shorter than predicted timelines

        NOTE
        -------------
        `infections` `hospitalization_preds`, and `hospitalization_ground_truth` are assumed to begin at
        parameters.config.INIT_DATE. Furthermore, it is assumed that the model predicts equal to or more
        days than is passed in `hospitalization_ground_truth`. If you pass in more ground truth data
        please cut it off on the last model prediction.

        Returns
        -------
        pd.DataFrame
            a pandas dataframe with a "date" column along with a number of views of the model output
        """
        num_days_predicted = infections[parameters.config.COMPARTMENT_IDX.S].shape[0]
        if hospitalization_ground_truth is not None:
            num_days_predicted = max()
        timeline = [
            utils.sim_day_to_date(day, parameters.config.INIT_DATE)
            for day in range(num_days_predicted)
        ]
        df = pd.DataFrame()
        df["date"] = timeline
        for age_bin_str in parameters.config.AGE_GROUP_STRS:
            age_bin_idx = parameters.config.AGE_GROUP_IDX[age_bin_str]
            if hospitalization_ground_truth is not None:
                df["obs_hosp_%s" % (age_bin_str.replace("-", "_"))] = (
                    hospitalization_ground_truth[:, age_bin_idx]
                )
        return df
