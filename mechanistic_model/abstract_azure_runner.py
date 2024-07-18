"""
The following abstract class defines a mechanistic_initializer,
which is used by mechanistic_runner instances to initialize the state on which ODEs will be run.
mechanistic_initializers will often be tasked with reading, parsing, and combining data sources
to produce an initial state representing some analyzed population
"""

import copy
import json
import os
import warnings
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from diffrax import Solution

import mechanistic_model.utils as utils
from mechanistic_model import SEIC_Compartments
from mechanistic_model.abstract_parameters import AbstractParameters
from mechanistic_model.mechanistic_inferer import MechanisticInferer
from mechanistic_model.static_value_parameters import StaticValueParameters


class AbstractAzureRunner(ABC):
    def __init__(self, azure_output_dir):
        # saving for future use
        self.azure_output_dir = azure_output_dir
        if not os.path.exists(azure_output_dir):
            os.makedirs(azure_output_dir, exist_ok=True)
        # create two dual loggers to save sys.stderr and sys.stdout to files in `azure_output_dir`
        self.dl_out = utils.dual_logger_out(
            os.path.join(azure_output_dir, "stdout.txt"), "w"
        )
        self.dl_err = utils.dual_logger_err(
            os.path.join(azure_output_dir, "stderr.txt"), "w"
        )

    @abstractmethod
    def process_state(self, state, **kwargs):
        """Abstract function meant to be implemented by the instance of the runner.
        This handles all of the logic of actually getting a solution object. Feel free to override
        or use a different function

        Calls upon save_config, save_inference_posteriors/save_static_run_timelines to
        easily save its outputs for later visualization.

        Parameters
        ----------
        state : str
            USPS state code for an individual state or territory.
        """
        pass

    def save_config(self, config_path: str, suffix: str = "_used"):
        """saves a config json located at `config_path` appending `suffix` to the filename
        to help distinguish it from other configs.

        Parameters
        ----------
        config_path : str
            the path, relative or absolute, to the config file wishing to be saved.
        suffix : str, optional
            suffix to append onto filename, if "" config path remains untouched, by default "_used"
        """
        config_path = config_path.replace(
            "\\", "/"
        )  # catches windows path weirdness
        # split extension and filename, then add suffix between
        # e.g. test_file.py -> test_file_used.py
        filename = os.path.basename(config_path)
        name, extension = os.path.splitext(filename)
        new_filename = f"{name}{suffix}{extension}"
        new_config_path = os.path.join(self.azure_output_dir, new_filename)
        with open(new_config_path, "w") as j:
            # open the location of `config_path` locally, and save it to the new_config_path
            json.dump(json.load(open(config_path, "r")), j, indent=4)

    def match_index_len(
        self, series: np.ndarray, index_len: int, pad: str = "l"
    ) -> np.ndarray:
        """A helper function designed to simply insert Nans on the left or right
        of a series so that it matches the desired `index_len`"""

        def _pad_fn(series, index_len, pad):
            if "l" in pad:
                return np.pad(
                    series,
                    (index_len - len(series), 0),
                    "constant",
                    constant_values=None,
                )
            elif "r" in pad:
                return np.pad(
                    series,
                    (0, index_len - len(series)),
                    "constant",
                    constant_values=None,
                )

        if len(series) < index_len:
            return _pad_fn(series, index_len, pad)
        return series

    def _generate_model_component_timelines(
        self,
        model: AbstractParameters,
        solution: Solution,
        hospitalization_preds: SEIC_Compartments = None,
    ) -> pd.DataFrame:
        """
        a private function which takes two timelines of infections and hospitalizations and generates a
        dataframe of different timelines of interest

        Parameters
        ----------
        parameters : AbstractParameters
            a class that inherits from AbstractParameters and therefore has a get_parameters() method
        solution : Solution
            diffrax Solution object as produced by MechanisticRunner.run() command
        vaccination_func : Optional Callable
            function used by `parameters` to generate vaccinations of
            each age bin x vax status strata. Takes input parameter `t` for day of sim
        seasonality_func : Optional Callable
            function used by `parameters` to generate seasonality coefficients.
            Takes input parameter `t` for day of sim
        external_i_func : Optional Callable
            function used by `parameters` to generate external introductions of
            each age bin x strain strata. Takes input parameter `t` for day of sim
        hospitalization_preds : Optional SEIC_Compartments
            models hospitalization predictions, usually the infections
            matrix with some infection hospitalization ratio applied

        NOTE
        -------------
        `infections` and `hospitalization_preds` are assumed to begin at
        parameters.config.INIT_DATE.

        Returns
        -------
        pd.DataFrame
            a pandas dataframe with a "date" column along with a number of views of the model output
        """
        infections = solution.ys
        num_days_predicted = infections[model.config.COMPARTMENT_IDX.S].shape[
            0
        ]
        timeline = [
            utils.sim_day_to_date(day, model.config.INIT_DATE)
            for day in range(num_days_predicted)
        ]
        df = pd.DataFrame()
        df["date"] = timeline
        parameters = model.get_parameters()
        vaccination_func = parameters["VACCINATION_RATES"]
        seasonality_func = parameters["SEASONALITY"]
        external_i_func = parameters["EXTERNAL_I"]
        # save a timeline of shape (num_days_predicted, age_groups)
        # sum across vax status
        vaccination_timeline = np.array(
            [
                np.sum(vaccination_func(t), axis=1)
                for t in range(num_days_predicted)
            ]
        )
        # save a seasonality timeline of shape (num_days_predicted, )
        seasonality_timeline = np.array(
            [seasonality_func(t) for t in range(num_days_predicted)]
        )
        df["seasonality_coef"] = seasonality_timeline
        # save external introductions timeline of shape (num_days_predicted, num_strains)
        # sum across age groups since we do % of each age bin anyways
        external_i_timeline = np.array(
            [
                np.sum(
                    external_i_func(t),
                    axis=(
                        model.config.I_AXIS_IDX.age,
                        model.config.I_AXIS_IDX.hist,
                        model.config.I_AXIS_IDX.vax,
                    ),
                )
                for t in range(num_days_predicted)
            ]
        )
        # select timeline of those with infect_hist 0, sum over all vax/wane tiers
        never_infected = np.sum(
            infections[model.config.COMPARTMENT_IDX.S][:, :, 0, :, :],
            axis=(model.config.S_AXIS_IDX.vax, model.config.S_AXIS_IDX.wane),
        )
        # 1-never_infected is those sero-positive / POPULATION to make proportions
        sim_sero = 1 - never_infected / model.config.POPULATION
        for age_bin_str in model.config.AGE_GROUP_STRS:
            age_bin_idx = model.config.AGE_GROUP_IDX[age_bin_str]
            age_bin_str = age_bin_str.replace("-", "_")
            # save ground truth and predicted hosp by age
            if hospitalization_preds is not None:
                df["pred_hosp_%s" % (age_bin_str)] = self.match_index_len(
                    hospitalization_preds[:, age_bin_idx], len(df.index)
                )
            # save vaccination rate by age
            df["vaccination_%s" % (age_bin_str)] = vaccination_timeline[
                :, age_bin_idx
            ]
            # save sero-positive rate by age
            df["sero_%s" % (age_bin_str)] = sim_sero[:, age_bin_idx]
        # get total infection incidence
        infection_incidence, _ = utils.get_timeline_from_solution_with_command(
            infections,
            model.config.COMPARTMENT_IDX,
            model.config.WANE_IDX,
            model.config.STRAIN_IDX,
            "incidence",
        )
        # incidence takes a diff, thus reducing length by 1
        # since we cant measure change in infections at t=0 we just prepend None
        # plots can start the next day.
        df["total_infection_incidence"] = self.match_index_len(
            infection_incidence, len(df.index)
        )
        # get strain proportion by strain for each day
        strain_proportions, _ = utils.get_timeline_from_solution_with_command(
            infections,
            model.config.COMPARTMENT_IDX,
            model.config.WANE_IDX,
            model.config.STRAIN_IDX,
            "strain_prevalence",
        )
        # shape (strain, num_days_predicted) summed across age bins
        population_immunity = np.mean(
            utils.get_immunity(model, solution), axis=-1
        )
        for s_idx, strain_name in enumerate(
            model.config.STRAIN_IDX._member_names_
        ):
            # save out strain specific data for each strain
            df["%s_strain_proportion" % strain_name] = strain_proportions[
                s_idx
            ]
            df[
                "%s_external_introductions" % strain_name
            ] = external_i_timeline[:, s_idx]
            df["%s_average_immunity" % strain_name] = population_immunity[
                s_idx, :
            ]
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
        else:
            warnings.warn(
                "attempting to call `save_inference_posteriors` before inference is complete. Something is likely wrong..."
            )

    def save_inference_timelines(
        self,
        inferer: MechanisticInferer,
        timeline_filename: str = "azure_visualizer_timeline.csv",
        particles_saved=1,
        extra_timelines: pd.DataFrame = None,
    ) -> str:
        """saves history of inferer sampled values for use by the azure visualizer.
        saves JSON file to `self.azure_output_dir/timeline_filename`.
        Look at `shiny_visualizers/azure_visualizer.py` for logic on parsing and visualizing the chains.
        Will error if inferer.infer() has not been run previous to this call.

        Parameters
        ----------
        inferer: MechanisticInferer
            the inferer object used to sample the parameter chains that will be visualized
        timeline_filename : str, optional
            filename to be saved under.
            DONT CHANGE WITHOUT MODIFICATION to `shiny_visualizers/azure_visualizer.py`, by default "azure_visualizer_timeline.csv"
        particles_saved : int, optional
            the number of particles per chain to save timelines for, by default 1
        extra_timelines: pd.DataFrame, optional
            a pandas dataframe containing a `date` column along with additional columns you wish
            to be recorded. Dates predicted by `inferer` not included in `extra_timelines` will
            be filled with `None`. `extra_timelines` are added identically to all `particles_saved`

        Returns
        -------
        str
            path the inference timelines were saved to
        """
        inference_visuals_save_path = os.path.join(
            self.azure_output_dir,
            timeline_filename,
        )
        assert (
            "date" in extra_timelines.columns
        ), "extra_timelines lacks a `date` column, "
        "can not be certain of when these observations occur"
        # attempt conversion to datetime column if it is not already
        try:
            extra_timelines["date"] = pd.to_datetime(extra_timelines["date"])
        except Exception as e:
            # we tried to cast `date` column to datetime and it failed, print and reraise error
            print(
                "Encountered an error trying to parse extra_timelines[date] into a datetime column"
            )
            raise e

        all_particles_df = pd.DataFrame()
        # randomly select `particles_saved` particles from the number of samples run
        for particle in np.random.choice(
            range(inferer.config.INFERENCE_NUM_SAMPLES),
            particles_saved,
            replace=False,
        ):
            # select this random particle for each of our chains
            chain_particle_pairs = [
                (chain, particle)
                for chain in range(inferer.config.INFERENCE_NUM_CHAINS)
            ]
            # assuming the inferer has finished fitting
            # load those particle posteriors and their solution dicts
            posteriors = inferer.load_posterior_particle(chain_particle_pairs)
            for (chain, particle), sol_dct in posteriors.items():
                # content of `sol_dct` depends on return value of inferer.likelihood func
                infection_timeline, hospitalizations, static_parameters = (
                    sol_dct["solution"],
                    sol_dct["hospitalizations"],
                    sol_dct["parameters"],
                )
                # spoof the inferer to return our static parameters when calling `get_parameters()`
                # instead of trying to sample like it normally does
                spoof_static_inferer = copy.copy(inferer)
                spoof_static_inferer.get_parameters = lambda: static_parameters
                df = self._generate_model_component_timelines(
                    spoof_static_inferer,
                    infection_timeline,
                    hospitalization_preds=hospitalizations,
                )
                df["chain_particle"] = "%s_%s" % (chain, particle)
                # add user specified extra timelines, filling in missing dates
                df = df.merge(extra_timelines, on="date", how="left")

                # add this chain/particle combo onto the main df
                all_particles_df = pd.concat(
                    [all_particles_df, df], axis=0, ignore_index=True
                )
        all_particles_df.to_csv(inference_visuals_save_path, index=False)
        return inference_visuals_save_path

    def save_static_run_timelines(
        self,
        parameters: StaticValueParameters,
        sol: Solution,
        timeline_filename: str = "azure_visualizer_timeline.csv",
    ):
        """given a tuple of compartment timelines, saves a number of timelines of interest for future visualization
        usually `sol` is retrieved from `diffrax.Solution.ys` which is an object returned from `MechanisticRunner.run()`

        Parameters
        ----------
        parameters : StaticValueParameters
            a version of AbstractParameters which is guaranteed to contain only static values.
            Otherwise use `save_inference_timelines`
        sol : Solution
            diffrax.Solution object returned from calling parameters.run()
        timeline_filename : str, optional
            filename to be saved under.
            DONT CHANGE WITHOUT MODIFICATION to `shiny_visualizers/azure_visualizer.py`,
            by default "azure_visualizer_timeline.csv"

        Returns
        -------
        str
            path the inference timelines were saved to
        """
        inference_visuals_save_path = os.path.join(
            self.azure_output_dir,
            timeline_filename,
        )
        # get a number of timelines to visualize the run
        df = self._generate_model_component_timelines(parameters, sol)
        # there is no chain nor particle in a static run, so we save as na_na
        df["chain_particle"] = "na_na"
        df.to_csv(inference_visuals_save_path, index=False)
        return inference_visuals_save_path
