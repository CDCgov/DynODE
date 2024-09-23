"""
a script which is passed two flags, one to dictate the experiment folder on which to run
and another to specify the location of the state-specific runner script, which will perform analysis on a single state.
"""

import argparse

from .azure_utilities import AzureExperimentLauncher

# specify job ID, cant already exist

DOCKER_IMAGE_TAG = "scenarios-image-7-3-24"
# number of seconds of a full experiment run before timeout
# for `s` states to run and `n` nodes dedicated,`s/n` * runtime 1 state secs needed
EXPERIMENTS_DIRECTORY = "exp"
SECRETS_PATH = "secrets/configuration_cfaazurebatchprd.toml"
# Parse command-line arguments
parser = argparse.ArgumentParser(description="Experiment Azure Launcher")
parser.add_argument(
    "--job_id",
    type=str,
    help="job ID of the azure job, must be unique",
    required=True,
)
parser.add_argument(
    "--experiment_name",
    type=str,
    help="the experiment name, must match the experiment directory within %s"
    % EXPERIMENTS_DIRECTORY,
    required=True,
    nargs="+",  # require at least 1 experiment
)
args = parser.parse_args()
experiment_names: list[str] = args.experiment_name
job_id: str = args.job_id
# upload dockerfile used
launcher = AzureExperimentLauncher(
    experiment_names[0],
    job_id,
    azure_config_toml=SECRETS_PATH,
    experiment_directory=EXPERIMENTS_DIRECTORY,
    docker_image_name=DOCKER_IMAGE_TAG,
)
launcher.set_resource_pool(pool_name="scenarios_4cpu_pool")
all_tasks_run = []
# all experiments will be placed under the same jobid,
# subsequent experiments depend on prior ones to finish before starting
for experiment_name in experiment_names:
    launcher.set_all_paths(
        experiments_folder_name=EXPERIMENTS_DIRECTORY,
        experiment_name=experiment_name,
    )
    state_task_ids = launcher.launch_states(depend_on_task_ids=all_tasks_run)
    postprocessing_tasks = launcher.launch_postprocess(
        depend_on_task_ids=state_task_ids
    )
    all_tasks_run += state_task_ids + postprocessing_tasks
launcher.azure_client.monitor_job(job_id)
