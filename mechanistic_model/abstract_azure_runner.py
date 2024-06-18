"""
The following abstract class defines a mechanistic_initializer,
which is used by mechanistic_runner instances to initialize the state on which ODEs will be run.
mechanistic_initializers will often be tasked with reading, parsing, and combining data sources
to produce an initial state representing some analyzed population
"""

import os
from abc import ABC, abstractmethod
from typing import Union
import json

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

    def _generate_model_component_timelines(
        self,
        parameters: AbstractParameters,
        infections: tuple[Array, Array, Array, Array],
        hospitalization_preds: tuple[Array, Array, Array, Array] = None,
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
        hospitalization_preds : Optional tuple[Array, Array, Array, Array]
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
            if hospitalization_preds is not None:
                df["pred_hosp_%s" % (age_bin_str.replace("-", "_"))] = (
                    hospitalization_preds[:, age_bin_idx]
                )
            infection_incidence = utils.get_timeline_from_solution_with_command(
                infections,
                parameters.config.COMPARTMENT_IDX,
                parameters.config.W_IDX,
                parameters.config.STRAIN_IDXs,
                "incidence",
            )
            df["total_infection_incidence"] = infection_incidence
            strain_proportions = utils.get_timeline_from_solution_with_command(
                infections,
                parameters.config.COMPARTMENT_IDX,
                parameters.config.W_IDX,
                parameters.config.STRAIN_IDXs,
                "strain_prevalence",
            )
            for s_idx, strain_name in enumerate(
                parameters.config.STRAIN_IDX._member_names_
            ):
                strain_infections = utils.get_timeline_from_solution_with_command(
                    infections,
                    parameters.config.COMPARTMENT_IDX,
                    parameters.config.W_IDX,
                    parameters.config.STRAIN_IDXs,
                    strain_name,
                )
                df["%s_exposed_infectious" % strain_name] = strain_infections
                df["%s_strain_proportion" % strain_name] = strain_proportions[s_idx]
        return df

    def save_inference_posteriors(
        self, inferer: MechanisticInferer, save_filename="checkpoint.json"
    ) -> None:
        """saves output of mcmc.get_samples(), does nothing if `inferer` has not compelted inference yet.

        Parameters
        ----------
        inferer : MechanisticInferer
            inferer that was run with `inferer.infer()`
        save_filename : str, optional
            output filename, by default "checkpoint.json"

        Returns
        ------------
        None
        """
        # if inference complete, convert jnp/np arrays to list, then json dump
        if inferer.infer_complete:
            samples = inferer.inference_algo.get_samples(group_by_chain=True)
            for param in samples.keys():
                samples[param] = samples[param].tolist()
            save_path = os.path.join(self.azure_output_dir, save_filename)
            json.dump(samples, open(save_path, "w"))

    def save_inference_visuals(
        self,
        inferer: MechanisticInferer,
        vis_filename: str = "azure_visualizer_timeline.csv",
        particles_saved=1,
    ) -> str:
        """saves history of inferer sampled values for use by the azure visualizer.
        saves JSON file to `self.azure_output_dir/vis_path`.
        Look at shiny_visualizers/azure_visualizer.py for logic on parsing and visualizing the chains.

        Parameters
        ----------
        inferer: MechanisticInferer
            the inferer object used to sample the parameter chains that will be visualized
        solution_interpreter : _type_
            _description_
        vis_filename : _type_, optional
            _description_, by default "azure_visualizer_inference_timeline.json"
        particles_run : _type_, optional
            _description_, by default 1

        Returns
        -------
        _type_
            _description_
        """
        inference_visuals_save_path = os.path.join(
            self.azure_output_dir,
            vis_filename,
        )
        all_particles_df = pd.DataFrame()
        for particle in range(particles_saved):
            # assuming the inferer has finished fitting
            posteriors = inferer.load_posterior_particle(particle)
            for (chain, particle), sol_dct in posteriors.items():
                infection_timeline, hospitalizations = (
                    sol_dct["solution"],
                    sol_dct["hospitalizations"],
                )
                df = self._generate_model_component_timelines(
                    inferer, infection_timeline, hospitalizations
                )
                df["chain_particle"] = "%s_%s" % (chain, particle)
                # add this chain/particle combo onto the main df
                all_particles_df = pd.concat(
                    [all_particles_df, df], axis=0, ignore_index=True
                )
        all_particles_df.to_csv(inference_visuals_save_path, index=False)
        return inference_visuals_save_path

    def save_single_run_visuals(
        self,
        parameters: AbstractParameters,
        sol: tuple[Array, Array, Array, Array],
        vis_filename: str = "azure_visualizer_timeline.png",
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
        df = self._generate_model_component_timelines(parameters, sol)
        df.to_csv(inference_visuals_save_path, index=False)
        return inference_visuals_save_path
