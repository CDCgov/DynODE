"""
The following abstract class defines a an abstract_azure_runner,
commonly used to accelerate runs of the model onto azure this file
aids the user in the production of timeseries to describe a model run

It also handles the saving of stderr and stdout copies as the job executes.
"""

import copy
import json
import os
import warnings
from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np
import pandas as pd  # type: ignore
from diffrax import Solution  # type: ignore
from jax import Array
from resp_ode import (
    AbstractParameters,
    MechanisticInferer,
    SEIC_Compartments,
    StaticValueParameters,
    utils,
    vis_utils,
)


class AbstractAzureRunner(ABC):
    """An Abstract class made to standardize the process of running an experiment on Azure.
    Children of this class may use the functions within to standardize their processies across experiments
    """

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
        self, series: Array, index_len: int, pad: str = "l"
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

    def _get_vaccination_timeseries(
        self, vaccination_func, num_days_predicted
    ) -> np.ndarray:
        """gets num individuals vaccinated by day and age bin

        Parameters
        ----------
        vaccination_func : Callable[int]
            function to generate vaccinations on a given day
        num_days_predicted : int
            number of days the simulation was run for

        Returns
        -------
        np.ndarray
            timeseries of shape (num_days_predicted, age) containing the
            output of vax_function applied on that day summed across all different vaccine stratifications
        """
        return np.array(
            [
                np.sum(vaccination_func(t), axis=1)
                for t in range(num_days_predicted)
            ]
        )

    def _get_seasonality_timeseries(
        self, seasonality_func, num_days_predicted: int
    ) -> np.ndarray:
        """gets seasonality coefficient by day

        Parameters
        ----------
        seasonality_func : Callable[int]
            function to generate seasonality coefficients in the model
        num_days_predicted : int
            number of days the simulation was run for

        Returns
        -------
        np.ndarray
            timeseries of shape (num_days_predicted,) containing the output of seasonality_func applied on that day
        """
        return np.array(
            [seasonality_func(t) for t in range(num_days_predicted)]
        )

    def _get_external_infection_timeseries(
        self,
        external_i_func,
        num_days_predicted: int,
        model: AbstractParameters,
    ) -> np.ndarray:
        """generates the external_introduction counts by day and strain

        Parameters
        ----------
        external_i_func : Callable[int]
            function to generate externally introduced people
        num_days_predicted : int
            number of days the simulation was run for
        model : AbstractParameters
            parameters object for enum lookup

        Returns
        -------
        np.ndarray
            timseries of all external introductions of shape (num_days_predicted, model.config.NUM_STRAINS)
        """
        # sum across age groups since we do % of each age bin anyways
        return np.array(
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

    def _get_sero_proportion_timeseries(
        self,
        compartment_timeseries: SEIC_Compartments,
        population: Array,
        model: AbstractParameters,
    ) -> np.ndarray:
        """given a timeseries of infections, finds individuals who
        have never been infected, and uses that to calculate the predicted sero-positive
        proportion among the population by day.

        Parameters
        ----------
        infections : SEIC_Compartments
            tuple of jax arrays containing a timeseries of each compartment's values
            on a given day.
        population : jax.Array
            an array of len(model.config.NUM_AGE_GROUPS) containing the
            population sizes of each age bin
        model : AbstractParameters
            a parameter object containing a config.S_AXIS_IDX enum for lookups and
            a config.POPULATION array for population counts

        Returns
        -------
        np.ndarray
            array of two dimensions (time, age) matching the first two dimensions
            of the Susceptible compartment's timeseries
        """
        # select timeline of those with infect_hist 0, sum over all vax/wane tiers
        never_infected = np.sum(
            compartment_timeseries[model.config.COMPARTMENT_IDX.S][
                :, :, 0, :, :
            ],
            axis=(model.config.S_AXIS_IDX.vax, model.config.S_AXIS_IDX.wane),
        )
        # 1-never_infected is those sero-positive / POPULATION to make proportions
        sim_sero = 1 - never_infected / population
        return sim_sero

    def _generate_model_component_timelines(
        self,
        model: AbstractParameters,
        solution: Solution,
        hospitalization_preds: Optional[Array] = None,
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
        hospitalization_preds : Optional Array
            models hospitalization predictions, usually the infections
            matrix with some infection hospitalization ratio applied,
            must be of shape (time, age_bin)

        NOTE
        -------------
        `infections` and `hospitalization_preds` are assumed to begin at
        parameters.config.INIT_DATE.

        Returns
        -------
        pd.DataFrame
            a pandas dataframe with a "date" column along with a number of views of the model output
        """
        compartment_timeseries = solution.ys
        num_days_predicted = compartment_timeseries[
            model.config.COMPARTMENT_IDX.S
        ].shape[0]
        timeline = [
            utils.sim_day_to_date(day, model.config.INIT_DATE)
            for day in range(num_days_predicted)
        ]
        df = pd.DataFrame()
        df["date"] = timeline
        parameters = model.get_parameters()
        # save a timeline of shape (num_days_predicted, age_groups)
        # sum across vax status
        vaccination_timeline = self._get_vaccination_timeseries(
            vaccination_func=parameters["VACCINATION_RATES"],
            num_days_predicted=num_days_predicted,
        )
        # save a seasonality timeline of shape (num_days_predicted, )
        df["seasonality_coef"] = self._get_seasonality_timeseries(
            seasonality_func=parameters["SEASONALITY"],
            num_days_predicted=num_days_predicted,
        )
        # save external introductions timeline of shape (num_days_predicted, num_strains)
        external_i_timeline = self._get_external_infection_timeseries(
            external_i_func=parameters["EXTERNAL_I"],
            num_days_predicted=num_days_predicted,
            model=model,
        )
        sim_sero = self._get_sero_proportion_timeseries(
            compartment_timeseries=compartment_timeseries,
            population=parameters["POPULATION"],
            model=model,
        )

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
            compartment_timeseries,
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
            compartment_timeseries,
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

    def _save_samples(self, samples, save_path):
        # convert np arrays to lists
        for param in samples.keys():
            samples[param] = samples[param].tolist()
        json.dump(samples, open(save_path, "w"))

    def save_mcmc_chains_plot(
        self,
        samples: dict[str : list : np.ndarray],
        save_filename: str = "mcmc_chains.png",
        plot_kwargs: dict = {},
    ):
        """Saves a plot mapping the MCMC chains of the inference job

        Parameters
        ----------
        samples : dict[str: list | np.ndarray]
            a dictionary (usually loaded from the checkpoint.json file) containing
            the sampled posteriors for each chain in the shape
            (num_chains, num_samples). All parameters generated with numpyro.plate
            and thus have a third dimension (num_chains, num_samples, num_plates)
            are flattened to the desired and displayed as
            separate parameters with _i suffix for each i in num_plates.
        save_filename : str, optional
            filename saved under, by default "mcmc_chains.png"
        plot_kwargs : dict, optional
            additional keyword arguments to pass to
            vis_utils.plot_mcmc_chains()
        """
        fig = vis_utils.plot_mcmc_chains(samples, **plot_kwargs)
        save_path = os.path.join(self.azure_output_dir, save_filename)
        fig.savefig(save_path, bbox_inches="tight")

    def save_correlation_pairs_plot(
        self,
        samples: dict[str : list : np.ndarray],
        save_filename: str = "mcmc_correlations.png",
        plot_kwargs: dict = {},
    ):
        """_summary_

        Parameters
        ----------
        samples : dict[str: list | np.ndarray]
            a dictionary (usually loaded from the checkpoint.json file) containing
            the sampled posteriors for each chain in the shape
            (num_chains, num_samples). All parameters generated with numpyro.plate
            and thus have a third dimension (num_chains, num_samples, num_plates)
            are flattened to the desired and displayed as
            separate parameters with _i suffix for each i in num_plates.
        save_filename : str, optional
            filename saved under, by default "mcmc_correlations.png"
        plot_kwargs : dict, optional
            additional keyword arguments to pass to
            vis_utils.plot_checkpoint_inference_correlation_pairs(),
            by default {}
        """
        fig = vis_utils.plot_checkpoint_inference_correlation_pairs(
            samples, **plot_kwargs
        )
        save_path = os.path.join(self.azure_output_dir, save_filename)
        fig.savefig(save_path, bbox_inches="tight")

    def save_inference_posteriors(
        self,
        inferer: MechanisticInferer,
        save_filename="checkpoint.json",
        exclude_prefixes=["final_timestep"],
        save_chains_plot=True,
        save_pairs_correlation_plot=True,
    ) -> None:
        """saves output of mcmc.get_samples(), does nothing if `inferer`
        has not compelted inference yet. By default saves accompanying
        visualizations for interpretability.

        Parameters
        ----------
        inferer : MechanisticInferer
            inferer that was run with `inferer.infer()`
        save_filename : str, optional
            output filename, by default "checkpoint.json"
        exclude_prefixes: list[str], optional
            a list of strs that, if found in a sample name,
            are exlcuded from the saved json. This is common for large logging
            info that will bloat filesize like, by default ["final_timestep"]
        save_chains_plot: bool, optional
            whether to save accompanying mcmc chains plot, by default True
        save_pairs_correlation_plot: bool, optional
            whether to save accompanying pairs correlation plot,
            by default True
        Returns
        ------------
        None
        """
        # if inference complete, convert jnp/np arrays to list, then json dump
        if inferer.infer_complete:
            samples: dict = inferer.inference_algo.get_samples(
                group_by_chain=True
            )
            # drop anything with a prefix found in exclude_prefixes
            samples = {
                name: posterior
                for name, posterior in samples.items()
                if not any([prefix in name for prefix in exclude_prefixes])
            }
            save_path = os.path.join(self.azure_output_dir, save_filename)
            self._save_samples(samples, save_path)
            # by default save an accompanying mcmc chains plot for readability
            if save_chains_plot:
                self.save_mcmc_chains_plot(samples)
            if save_pairs_correlation_plot:
                self.save_correlation_pairs_plot(samples)
        else:
            warnings.warn(
                "attempting to call `save_inference_posteriors` before inference is complete. Something is likely wrong..."
            )

    def save_inference_final_timesteps(
        self,
        inferer: MechanisticInferer,
        save_filename="final_timesteps.json",
        final_timestep_identifier="final_timestep",
    ):
        """saves the `final_timestep` posterior, if it is found in mcmc.get_samples(), otherwise raises a warning
        and saves nothing

        Parameters
        ----------
        inferer : MechanisticInferer
            inferer that was run with `inferer.infer()`
        save_filename : str, optional
            output filename, by default "final_timesteps.json"
        final_timestep_identifier : str, optional
            prefix attached to the final_timestep parameter, by default "final_timestep"
        """
        # if inference complete, convert jnp/np arrays to list, then json dump
        if inferer.infer_complete:
            samples = inferer.inference_algo.get_samples(group_by_chain=True)
            final_timesteps = {
                name: timesteps
                for name, timesteps in samples.items()
                if final_timestep_identifier in name
            }
            # if it is empty, warn the user, save nothing
            if final_timesteps:
                save_path = os.path.join(self.azure_output_dir, save_filename)
                self._save_samples(final_timesteps, save_path)
            else:
                warnings.warn(
                    "attempting to call `save_inference_final_timesteps` but failed to find any final_timesteps with prefix %s"
                    % final_timestep_identifier
                )
        else:
            warnings.warn(
                "attempting to call `save_inference_final_timesteps` before inference is complete. Something is likely wrong..."
            )

    def save_inference_timelines(
        self,
        inferer: MechanisticInferer,
        timeline_filename: str = "azure_visualizer_timeline.csv",
        particles_saved=1,
        extra_timelines: pd.DataFrame = None,
        tf: Union[int, None] = None,
        external_particle: dict[str, Array] = {},
    ) -> str:
        """saves history of inferer sampled values for use by the azure visualizer.
        saves CSV file to `self.azure_output_dir/timeline_filename`.
        Look at `shiny_visualizers/azure_visualizer.py` for logic on parsing and visualizing the chains.
        Will error if inferer.infer() has not been run previous to this call or an external_particle is passed.

        Parameters
        ----------
        inferer: MechanisticInferer
            the inferer object used to sample the parameter chains that will be visualized
        timeline_filename : str, optional
            filename to be saved under.
            DONT CHANGE WITHOUT MODIFICATION to downstream postprocessing scripts, by default "azure_visualizer_timeline.csv"
        particles_saved : int, optional
            the number of particles per chain to save timelines for, by default 1
        extra_timelines: pd.DataFrame, optional
            a pandas dataframe containing a `date` column along with additional columns you wish
            to be recorded. Dates predicted by `inferer` not included in `extra_timelines` will
            be filled with `None`. `extra_timelines` are added identically to all `particles_saved`
        tf: Union[int, None]:
            number of days to run posterior model for, defaults to same number of days used in fitting
            if possible.
        external_posteriors: dict
            for use of particles defined somewhere outside of this instance of the MechanisticInferer.
            For example, loading a checkpoint.json containing saved posteriors from an Azure Batch job.
            expects keys that match those given to `numpyro.sample` often from
            inference_algo.get_samples(group_by_chain=True).

        Returns
        -------
        str
            path the inference timelines were saved to
        """
        inference_visuals_save_path = os.path.join(
            self.azure_output_dir,
            timeline_filename,
        )
        if isinstance(extra_timelines, pd.DataFrame):
            assert (
                "date" in extra_timelines.columns
            ), "extra_timelines lacks a `date` column, "
            "can not be certain of when these observations occur"
            # attempt conversion to datetime column if it is not already
            try:
                extra_timelines["date"] = pd.to_datetime(
                    extra_timelines["date"]
                )
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
            posteriors = inferer.load_posterior_particle(
                chain_particle_pairs,
                tf=tf,
                external_particle=external_particle,
            )
            for (chain, particle), sol_dct in posteriors.items():
                # content of `sol_dct` depends on return value of inferer.likelihood func
                infection_timeline: Solution = sol_dct["solution"]
                hospitalizations: Array = sol_dct["hospitalizations"]
                static_parameters: dict[str, Array] = sol_dct["parameters"]
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
                if isinstance(extra_timelines, pd.DataFrame):
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
