"""
a script which is passed two flags, one to dictate the experiment folder on which to run
and another to specify the location of the state-specific runner script, which will perform analysis on a single state.
"""

import argparse
import os

from src.mechanistic_azure.azure_utilities import AzureExperimentLauncher

# from cfa_azure.clients import AzureClient


class ScenarioLauncher(AzureExperimentLauncher):
    """A class designed to launch jobs but specifically a number of scenarios
    per state.
    """

    def launch_states(self, depend_on_task_ids: list[str] = None) -> list[str]:
        """
        OVERIDDEN FUNCTION to launch scenarios for each state as well as the states themselves
        Launches an Azure Batch job under `self.job_id`,
        populating it with tasks for each subdirectory within your experiment's `states` directory
        passing each state name to `run_task.py` with the -s flag and the job_id with the -j flag.

        Parameters
        ----------
        depend_on_task_ids: list[str], optional
            list of task ids on which each state depends on finishing to start themselves, defaults to None
        Returns
        -------
        list[str]
            list of all tasks launched under `job_id`
        """
        # command to run the job, if we have already launched states previously, dont recreate the job
        if not self.job_launched:
            self.azure_client.add_job(job_id=self.job_id)
            self.job_launched = True
        # upload the experiment folder so that the runner_path_docker & states_path_docker point to the correct places
        # here we pass `location=self.experiment_path_blob` because we are not uploading from the docker container
        # therefore we dont need the /input/ mount directory
        self._upload_experiment_to_blob()
        task_ids = []
        # add a task for each scenario json in state directory in the states folder of this experiment
        for statedir in os.listdir(self.states_path_local):
            statedir_path = os.path.join(self.states_path_local, statedir)
            if os.path.isdir(statedir_path):
                # add a task setting the runner onto each state
                # we use the -s flag with the subdir name,
                # since experiment directories are structured with USPS state codes as directory names
                # also include the -j flag to specify the jobid
                scenarios = [
                    f.replace(".json", "")
                    for f in os.listdir(statedir_path)
                    if "config_global" not in f
                ]
                for sc in scenarios:
                    task_id = self.azure_client.add_task(
                        job_id=job_id,
                        docker_cmd="python %s -s %s -j %s -sc %s"
                        % (self.runner_path_docker, statedir, job_id, sc),
                        depends_on=depend_on_task_ids,
                    )
                    task_ids += task_id
        return task_ids


DOCKER_IMAGE_TAG = "arik-revamp-projections1"
# number of seconds of a full experiment run before timeout
# for `s` states to run and `n` nodes dedicated,`s/n` * runtime 1 state secs needed
TIMEOUT_MINS = 360
EXPERIMENTS_DIRECTORY = "exp"
EXPERIMENT_NAME = "projections_ihr2_2404_2507"
SECRETS_PATH = "secrets/configuration_cfaazurebatchprd.toml"
# Parse command-line arguments
# specify job ID, cant already exist
parser = argparse.ArgumentParser(description="Experiment Azure Launcher")
parser.add_argument(
    "--job_id",
    type=str,
    help="job ID of the azure job, must be unique",
    required=True,
)

args = parser.parse_args()
job_id: str = args.job_id
launcher = ScenarioLauncher(
    EXPERIMENT_NAME,
    job_id,
    azure_config_toml=SECRETS_PATH,
    experiment_directory=EXPERIMENTS_DIRECTORY,
    docker_image_name=DOCKER_IMAGE_TAG,
)
launcher.set_resource_pool(pool_name="scenarios_4cpu_pool")
all_tasks_run = []
# all experiments will be placed under the same jobid,
# subsequent experiments depend on prior ones to finish before starting
launcher.set_all_paths(
    experiments_folder_name=EXPERIMENTS_DIRECTORY,
    experiment_name=EXPERIMENT_NAME,
)
state_task_ids = launcher.launch_states(depend_on_task_ids=all_tasks_run)
print(state_task_ids)
postprocessing_tasks = launcher.launch_postprocess(
    depend_on_task_ids=state_task_ids
)
all_tasks_run += state_task_ids + postprocessing_tasks
launcher.azure_client.monitor_job(job_id)
