"""
a script which is passed two flags, one to dictate the experiment folder on which to run
and another to specify the location of the state-specific runner script, which will perform analysis on a single state.
"""

import argparse
import os

from cfa_azure.clients import AzureClient

# specify job ID, cant already exist

<<<<<<< HEAD
DOCKER_IMAGE_TAG = "ben-scenarios-20240715"
# number of seconds of a full experiment run before timeout
# for `s` states to run and `n` nodes dedicated,`s/n` * runtime 1 state secs needed
TIMEOUT_MINS = 240
EXPERIMENTS_DIRECTORY = "exp"
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
)
args = parser.parse_args()
experiment_name: str = args.experiment_name
job_id: str = args.job_id
# using the experiment name, get the local (this machine) and docker (azure batch node) paths to each file
# NOTE: we prepend /input/ to the docker path since our azure blob is MOUNTED onto a docker container
# and the mount appears as a folder /input
folder_path_local = os.path.join(EXPERIMENTS_DIRECTORY, experiment_name)
folder_path_docker = os.path.join(
    "/input/", EXPERIMENTS_DIRECTORY, experiment_name, job_id
)
# get a path to the run_task python script, both on this machine and in the docker blob
runner_path_local = os.path.join(folder_path_local, "run_task.py")
runner_path_docker = os.path.join(folder_path_docker, "run_task.py")
# get a path to the `states` folder, both on this machine and in the docker blob
states_path_local = os.path.join(folder_path_local, "states")
states_path_docker = os.path.join(folder_path_docker, "states")
# check that the files are in the right place locally
assert os.path.exists(
    runner_path_local
), "make sure your run_task.py is found inside of your experiment folder"
assert os.path.exists(
    states_path_local
), "please rerun this file specifying the experiment directory, "
"it must include a /states/ folder within it containing the state folders filled with configs"
# start up azure client with authentication toml
client = AzureClient(config_path="secrets/configuration_cfaazurebatchprd.toml")
client.timeout = TIMEOUT_MINS
# run `docker build` using the Dockerfile in the cwd, apply tag
client.package_and_upload_dockerfile(
    registry_name="cfaprdbatchcr", repo_name="scenarios", tag=DOCKER_IMAGE_TAG
)
# create the input and output blobs, for now they must be named /input and /output
client.set_input_container("scenarios-mechanistic-input", "input")
client.set_output_container("scenarios-mechanistic-output", "output")

# upload the experiment folder so that the runner_path_docker & states_path_docker point to the correct places
# here we pass `location=exp/experiment_name/job_id` WITHOUT the /input/ folder because
# we are uploading directly to the blob, and not through the docker container
client.upload_files_in_folder(
    [folder_path_local],
    "scenarios-mechanistic-input",
    location="%s/%s/%s" % (EXPERIMENTS_DIRECTORY, experiment_name, job_id),
)

# IF CREATING A NEW POOL UNCOMMENT THE NEXT TWO LINES
# client.set_pool_info(
#     mode="autoscale",
#     autoscale_formula_path="secrets/autoscale.txt",
#     timeout=TIMEOUT_MINS,
# )
# client.create_pool(pool_name="scenarios_8cpu_pool_new")
# TO USE EXISTING POOL USE THIS LINE
client.set_pool("scenarios_8cpu_pool_new")

# command to run the job
client.add_job(job_id=job_id)
# add a task for each state directory in the states folder of this experiment
for statedir in os.listdir(states_path_local):
    statedir_path = os.path.join(states_path_local, statedir)
    if os.path.isdir(statedir_path):
        # add a task setting the runner onto each state
        # we use the -s flag with the subdir name,
        # since experiment directories are structured with USPS state codes as directory names
        # also include the -j flag to specify the jobid
        client.add_task(
            job_id=job_id,
            docker_cmd="python %s -s %s -j %s"
            % (runner_path_docker, statedir, job_id),
        )
client.monitor_job(job_id=job_id)
