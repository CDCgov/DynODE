"""Defines a an abstract_azure_runner, to standardize DynODE experiments.

Commonly used to accelerate runs of the model onto azure this file
aids the user in the production of timeseries to describe a model run

It also handles the saving of stderr and stdout copies as the job executes.
"""

import json
import os
import warnings
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd  # type: ignore
from diffrax import Solution  # type: ignore
from jax import Array

from . import utils, vis_utils
from .mechanistic_inferer import MechanisticInferer
from .static_value_parameters import StaticValueParameters


class AbstractDynodeRunner(ABC):
    """An abstract class made to standardize the process of running simulations and fitting.

    Children of this class may use functions within to standardize their processies across experiments.
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
        """Abstract function meant to be implemented by instance of the runner.

        Entry point that handles all of the logic of getting a solution object.

        Should call helper functions like `save_*` methods to
        easily save its outputs for later visualization.

        Parameters
        ----------
        state : str
            USPS state code for an individual state or territory.
        kwargs : any
            any other parameters needed to identify an individual simulation.
        """
        pass

    def save_config(self, config_path: str, suffix: str = "_used"):
        """Save a copy of config json located at `config_path`.

        Appends `suffix` to the filename to help distinguish it from input configs.

        Parameters
        ----------
        config_path : str
            the path, relative or absolute,
            to the config file wishing to be saved.
        suffix : str, optional
            suffix to append onto filename,
            if "" config filename remains untouched, by default "_used"
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
        samples: dict[str, list | np.ndarray],
        save_filename: str = "mcmc_chains.png",
        plot_kwargs: dict = {},
    ):
        """Save plot of posterior sample values by chain.

        Parameters
        ----------
        samples : dict[str, list | np.ndarray]
            Sampled posteriors for each chain in the shape
            (num_chains, num_samples).
        save_filename : str, optional
            filename saved under, by default "mcmc_chains.png"
        plot_kwargs : dict, optional
            additional keyword arguments to pass to
            vis_utils.plot_mcmc_chains()

        Notes
        -----
        All parameters generated with numpyro.plate within `samples`
        and thus have a third dimension (num_chains, num_samples, num_plates)
        are flattened to the desired and displayed as
        separate parameters with _i suffix for each i in num_plates.

        Most often `samples` is generated via `MCMC.get_samples()` on a
        finished inference object.
        """
        fig = vis_utils.plot_mcmc_chains(samples, **plot_kwargs)
        save_path = os.path.join(self.azure_output_dir, save_filename)
        fig.savefig(save_path, bbox_inches="tight")

    def save_correlation_pairs_plot(
        self,
        samples: dict[str, list | np.ndarray],
        save_filename: str = "mcmc_correlations.png",
        plot_kwargs: dict = {},
    ):
        """Save correlation pairs plot of sampled parameters.

        Parameters
        ----------
        samples : dict[str, list | np.ndarray]
            Sampled posteriors for each chain in the shape
            (num_chains, num_samples).
        save_filename : str, optional
            filename saved under, by default "mcmc_correlations.png"
        plot_kwargs : dict, optional
            additional keyword arguments to pass to
            vis_utils.plot_checkpoint_inference_correlation_pairs(),
            by default {}

        Notes
        -----
        All parameters generated with numpyro.plate within `samples`
        and thus have a third dimension (num_chains, num_samples, num_plates)
        are flattened to the desired and displayed as
        separate parameters with _i suffix for each i in num_plates.

        Most often `samples` is generated via `MCMC.get_samples()` on a
        finished inference object.
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
        """Save samples generated by `inferer` with accompanying visualizations.

        Parameters
        ----------
        inferer : MechanisticInferer
            Inferer on which inference was performed via `inferer.infer()`
        save_filename : str, optional
            Output filename, by default "checkpoint.json"
        exclude_prefixes: list[str], optional
            Prefixes to samples to exclude from `save_filename` for memory reasons.
            by default ["timestep"] to exclude all timestep logging variables.
        save_chains_plot: bool, optional
            Whether to save accompanying mcmc chains plot, by default True
        save_pairs_correlation_plot: bool, optional
            Whether to save accompanying pairs correlation plot,
            by default True
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
                "attempting to call `save_inference_posteriors` before "
                "inference is complete. Something is likely wrong..."
            )

    def save_inference_final_timesteps(
        self,
        inferer: MechanisticInferer,
        save_filename="final_timesteps.json",
    ):
        """Save each posterior sample's final compartment sizes.

        If no `final_timestep_*` samples found raises a warning and saves nothing.

        Parameters
        ----------
        inferer : MechanisticInferer
            Inferer that was run with `inferer.infer()`
        save_filename : str, optional
            Output filename, by default "timesteps.json"
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
        """Save compartment sizes at specified times for each posterior sample.

        Raises a warning and saves nothing if `timestep_identifier` numpyro
        sites are not found.

        Parameters
        ----------
        inferer : MechanisticInferer
            inferer that was run with `inferer.infer()`
        save_filename : str, optional
            output filename, by default "timesteps.json"
        step_identifier : str, optional
            identifying token attached to any timestep parameter,
            by default "timestep".
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
        particles: list[tuple[int, int]],
        timeseries_filename: str = "simulation_timeseries.csv",
        extra_timeseries: None | pd.DataFrame = None,
        tf: int | None = None,
        external_particle: dict[str, Array] = {},
        verbose: bool = False,
    ) -> pd.DataFrame:
        """Save timeseries of metrics for a number of posterior samples.

        Metrics of interest saved include vaccination, infections,
        variant proportions.


        Parameters
        ----------
        inferer: MechanisticInferer
            The inferer object used to sample the parameter chains that will be visualized.
        particles: list[tuple[int]]
            list of (chain, particle) tuples identifying which posteriors you
            wish to save, values must be less than
            `model.config.INFERENCE_NUM_CHAINS/SAMPLES` respectively.
        timeseries_filename : str, optional
            Filename to be saved under. None does not save file.
            Do not change without modification to downstream postprocessing
            scripts, by default "simulation_timeseries.csv"
        extra_timeseries: pd.DataFrame, optional
            A pandas dataframe containing a `date` column with additional
            metrics to be recorded. Dates predicted by `inferer` not included
            in `extra_timeseries` will be filled with `None`.
            `extra_timeseries` are joined identically to all `particles_saved`.
        tf: Union[int, None]:
            Number of days to run posterior model for,
            defaults to same number of days used in fitting if possible.
        external_posteriors: dict
            Posterior samples to use instead of `inferer.get_samples()`.
            Optional, default = {}.
        verbose: bool, optional
            Print out the current chain_particle value being executed.

        Returns
        -------
        pd.DataFrame
            dataframe containing all particles and their corresponding timeseries.

        Notes
        -----
        If the model passed is StaticValueParameters,
        all inference logic for choosing posterior particles is ignored and a
        single "particle" is simulated.

        Inferer does not need to be the same object that completed inference.
        For example, loading a checkpoint.json containing saved posteriors
        from a previous job then passing that dictionary to `external_particle`
        will produce identical results.
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
            hospitalizations_tmp = sol_dct["hospitalizations"]
            assert isinstance(hospitalizations_tmp, Array)
            hospitalizations: Array = hospitalizations_tmp
            parameters_tmp = sol_dct["parameters"]
            assert isinstance(parameters_tmp, dict)
            static_parameters: dict[str, Array] = parameters_tmp
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
        """Save timeseries of metrics for each timestep simulated.

        For use instead of `save_inference_timeseries` when dealing with
        static parameters only and no inference.

        Parameters
        ----------
        parameters : StaticValueParameters
            A version of AbstractParameters which is guaranteed to contain
            only static values. Otherwise use `save_inference_timeseries`
        sol : Solution
            diffrax.Solution object returned from calling parameters.run()
        timeseries_filename : str, optional
            Filename to be saved under. None does not save file.
            Do not change without modification to downstream postprocessing
            scripts, by default "simulation_timeseries.csv"

        Returns
        -------
        pd.DataFrame
            Dataframe containing timeseries of metrics of note.
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
        self, extra_timeseries: pd.DataFrame | None
    ) -> pd.DataFrame | None:
        """Validate the extra timeseries DataFrame."""
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
        self, particles: list[tuple[int, int]], inferer: MechanisticInferer
    ):
        """Validate the provided particles against configuration limits."""
        for particle in particles:
            (chain_num, particle_num) = particle

            assert (chain_num < inferer.config.INFERENCE_NUM_CHAINS) and (
                particle_num < inferer.config.INFERENCE_NUM_SAMPLES
            ), (
                f"Selected (chain, particle) ({chain_num}, {particle_num}) is out of range "
                f"of ({inferer.config.INFERENCE_NUM_CHAINS}, {inferer.config.INFERENCE_NUM_SAMPLES})."
            )


class _spoof_static_inferer(MechanisticInferer):
    """A spoof class made return static params instead of sampling."""

    def __init__(self, inferer, static_params):
        self.__dict__ = inferer.__dict__
        self.static_params = static_params

    def get_parameters(self):
        return self.static_params
