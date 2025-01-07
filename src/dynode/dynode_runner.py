"""
The following abstract class defines a an abstract_azure_runner,
commonly used to accelerate runs of the model onto azure this file
aids the user in the production of timeseries to describe a model run

It also handles the saving of stderr and stdout copies as the job executes.
"""

import json
import os
import warnings
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pandas as pd  # type: ignore
from diffrax import Solution  # type: ignore
from jax import Array

from . import utils, vis_utils
from .mechanistic_inferer import MechanisticInferer
from .static_value_parameters import StaticValueParameters


class AbstractDynodeRunner(ABC):
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

        Calls upon `save_*` methods to easily save its outputs for later visualization.

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
        exclude_prefixes=["timestep"],
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
            info that will bloat filesize like, by default ["timestep"]
            to exclude all timestep deterministic variables.
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
            utils.save_samples(samples, save_path, indent=0)
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
    ):
        """saves the `final_timestep` posterior, if it is found in
        mcmc.get_samples(), otherwise raises a warning and saves nothing

        Parameters
        ----------
        inferer : MechanisticInferer
            inferer that was run with `inferer.infer()`
        save_filename : str, optional
            output filename, by default "timesteps.json"
        final_timestep_identifier : str, optional
            prefix attached to the final_timestep parameter, by default "timestep"
        """
        self.save_inference_timesteps(
            inferer, save_filename, timestep_identifier="final_timestep"
        )

    def save_inference_timesteps(
        self,
        inferer: MechanisticInferer,
        save_filename="timesteps.json",
        timestep_identifier="timestep",
    ):
        """saves all `timestep` posteriors, if they are found in
        mcmc.get_samples(), otherwise raises a warning and saves nothing

        Parameters
        ----------
        inferer : MechanisticInferer
            inferer that was run with `inferer.infer()`
        save_filename : str, optional
            output filename, by default "timesteps.json"
        step_identifier : str, optional
            identifying token attached to any timestep parameter, by default "timestep"
        """
        # if inference complete, convert jnp/np arrays to list, then json dump
        if inferer.infer_complete:
            samples = inferer.inference_algo.get_samples(group_by_chain=True)
            timesteps = {
                name: timesteps
                for name, timesteps in samples.items()
                if timestep_identifier in name
            }
            # if it is empty, warn the user, save nothing
            if timesteps:
                save_path = os.path.join(self.azure_output_dir, save_filename)
                utils.save_samples(timesteps, save_path)
            else:
                warnings.warn(
                    "attempting to call `save_inference_timesteps` but failed to find any timesteps with prefix %s"
                    % timestep_identifier
                )
        else:
            warnings.warn(
                "attempting to call `save_inference_timesteps` before inference is complete. Something is likely wrong..."
            )

    def save_inference_timeseries(
        self,
        inferer: MechanisticInferer,
        particles: list[tuple[int]],
        timeseries_filename: str = "simulation_timeseries.csv",
        extra_timeseries: pd.DataFrame = None,
        tf: Union[int, None] = None,
        external_particle: dict[str, Array] = {},
        verbose: bool = False,
    ) -> pd.DataFrame:
        """saves model timeseries of important metrics of interest such as
        vaccination, infections, variant proportions etc for a given
        simulation period. If the model passed is StaticValueParameters,
        all inference logic for choosing posterior particles is ignored and a
        single "particle" is simulated.

        Parameters
        ----------
        inferer: MechanisticInferer
            the inferer object used to sample the parameter chains that will be visualized
        particles: list[tuple[int]]
            list of (chain, particle) tuples identifying which posteriors you
            wish to save, values must be less than
            `model.config.INFERENCE_NUM_CHAINS/SAMPLES` respectively.
        timeseries_filename : str, optional
            filename to be saved under. None does not save file.
            DONT CHANGE WITHOUT MODIFICATION to downstream postprocessing
            scripts, by default "simulation_timeseries.csv"
        extra_timeseries: pd.DataFrame, optional
            a pandas dataframe containing a `date` column along with additional columns you wish
            to be recorded. Dates predicted by `inferer` not included in `extra_timeseries` will
            be filled with `None`. `extra_timeseries` are added identically to all `particles_saved`
        tf: Union[int, None]:
            number of days to run posterior model for, defaults to same number of days used in fitting
            if possible.
        external_posteriors: dict
            for use of particles defined somewhere outside of this instance of the MechanisticInferer.
            For example, loading a checkpoint.json containing saved posteriors from an Azure Batch job.
            expects keys that match those given to `numpyro.sample` often from
            inference_algo.get_samples(group_by_chain=True).
        verbose: bool, optional
            whether or not to pring out the current chain_particle value being executed

        Returns
        -------
        pd.DataFrame
            dataframe containing all particles and their corresponding timeseries.

        """
        # Validate inputs before proceeding with main logic.
        extra_timeseries = self._validate_extra_timeseries(extra_timeseries)
        self._validate_particles(particles, inferer)

        all_particles_df = pd.DataFrame()
        # lookup each posterior particle and run simulation with that posterior
        posteriors = inferer.load_posterior_particle(
            particles,
            tf=tf,
            external_particle=external_particle,
            verbose=verbose,
        )
        for (chain, particle), sol_dct in posteriors.items():
            # content of `sol_dct` depends on return value of inferer.run_simulation func
            infection_timeseries: Solution = sol_dct["solution"]
            hospitalizations: Array = sol_dct["hospitalizations"]
            static_parameters: dict[str, Array] = sol_dct["parameters"]
            # spoof the inferer to return our static parameters when calling `get_parameters()`
            # instead of trying to sample like it normally does
            spoof_static_inferer = _spoof_static_inferer(
                inferer, static_parameters
            )
            df = utils.generate_model_component_timeseries(
                spoof_static_inferer,
                infection_timeseries,
                hospitalization_preds=hospitalizations,
            )
            df["chain_particle"] = "%s_%s" % (chain, particle)
            # add user specified extra timeseriess, filling in missing dates
            if isinstance(extra_timeseries, pd.DataFrame):
                df = df.merge(extra_timeseries, on="date", how="left")

            # add this chain/particle combo onto the main df
            all_particles_df = pd.concat(
                [all_particles_df, df], axis=0, ignore_index=True
            )
        # if not none, save csv
        if timeseries_filename:
            all_particles_df.to_csv(
                os.path.join(
                    self.azure_output_dir,
                    timeseries_filename,
                ),
                index=False,
            )
        return all_particles_df

    def save_static_run_timeseries(
        self,
        parameters: StaticValueParameters,
        sol: Solution,
        timeseries_filename: str = "simulation_timeseries.csv",
    ) -> pd.DataFrame:
        """given a tuple of compartment timeseries, saves a number of timeseries of interest for future visualization
        usually `sol` is retrieved from `diffrax.Solution.ys` which is an object returned from `MechanisticRunner.run()`

        Parameters
        ----------
        parameters : StaticValueParameters
            a version of AbstractParameters which is guaranteed to contain only static values.
            Otherwise use `save_inference_timeseries`
        sol : Solution
            diffrax.Solution object returned from calling parameters.run()
        timeseries_filename : str, optional
            filename to be saved under.
            DONT CHANGE WITHOUT MODIFICATION to `shiny_visualizers/azure_visualizer.py`,
            by default "simulation_timeseries.csv"

        Returns
        -------
        pd.DataFrame
            dataframe containing timeseries of metrics of note.
        """
        df = utils.generate_model_component_timeseries(parameters, sol)
        # there is no chain nor particle in a static run, so we save as na_na
        df["chain_particle"] = "na_na"
        if timeseries_filename:
            df.to_csv(
                os.path.join(
                    self.azure_output_dir,
                    timeseries_filename,
                ),
                index=False,
            )
        return df

    def _validate_extra_timeseries(
        self, extra_timeseries: pd.DataFrame
    ) -> pd.DataFrame:
        """Validates the extra timeseries DataFrame."""
        if isinstance(extra_timeseries, pd.DataFrame):
            assert (
                "date" in extra_timeseries.columns
            ), "extra_timeseries lacks a 'date' column; cannot determine when these observations occur."
            try:
                extra_timeseries["date"] = pd.to_datetime(
                    extra_timeseries["date"]
                )
            except Exception as e:
                print(
                    "Encountered an error trying to parse extra_timeseries['date'] into a datetime column."
                )
                raise e
        return extra_timeseries

    def _validate_particles(
        self, particles: list[tuple[int]], inferer: MechanisticInferer
    ):
        """Validates the provided particles against configuration limits."""
        for particle in particles:
            chain_num, particle_num = particle

            assert (chain_num < inferer.config.INFERENCE_NUM_CHAINS) and (
                particle_num < inferer.config.INFERENCE_NUM_SAMPLES
            ), (
                f"Selected (chain, particle) ({chain_num}, {particle_num}) is out of range "
                f"of ({inferer.config.INFERENCE_NUM_CHAINS}, {inferer.config.INFERENCE_NUM_SAMPLES})."
            )


class _spoof_static_inferer(MechanisticInferer):
    """A super simple spoof class made to mimic the current state of an
    inferer object but without resampling the parameters, instead
    returning a pre-specified dictionary `static_params`
    """

    def __init__(self, inferer, static_params):
        self.__dict__ = inferer.__dict__
        self.static_params = static_params

    def get_parameters(self):
        return self.static_params
