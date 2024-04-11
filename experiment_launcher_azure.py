"""
a script which is passed two flags, one to dictate the experiment folder on which to run
and another to specify the location of the state-specific runner script, which will perform analysis on a single state.
"""

import argparse
import os

from cfa_azure.clients import AzureClient

# specify job ID, cant already exist
JOB_ID = "scenarios_inference_run_14"
# number of seconds of a full experiment run before timeout
# for `s` states to run and `n` nodes dedicated,`s/n` * runtime 1 state secs needed
TIMEOUT_MINS = 120
# Parse command-line arguments
parser = argparse.ArgumentParser(description="Experiment Azure Launcher")
parser.add_argument(
    "--folder",
    type=str,
    help="Directory of experiment",
    required=True,
)
parser.add_argument(
    "--runner",
    type=str,
    help="Path to single state runner Python file INSIDE THE CONTAINER, prepend /app/",
    required=True,
)
args = parser.parse_args()
client = AzureClient(config_path="secrets/configuration_cfaazurebatchprd.toml")
# optional debugging, may mess with ability to set scalings
# client.set_debugging(True)
# run `docker build` using the Dockerfile in the cwd, apply tag
client.package_and_upload_dockerfile(
    registry_name="cfaprdbatchcr", repo_name="scenarios", tag="expinfer3"
)
# create the input and output blobs, for now they must be named /input and /output
client.set_input_container("scenarios-test-container", "input")
client.set_output_container("example-output-scenarios-mechanistic", "output")

# set the scaling of the pool, assign `dedicated_nodes` to split work accross
client.set_scaling(mode="fixed", dedicated_nodes=5, timeout=TIMEOUT_MINS)
# create the pool
client.create_pool(pool_name="scenarios_5_node")
# or use a certain pool if already exists and active
# client.use_pool(pool_name="scenarios_2_node")

# command to run the job
client.add_job(job_id=JOB_ID)
# add a task for each subdir of the given experiment folder
for subdir in os.listdir(args.folder):
    subdir_path = os.path.join(args.folder, subdir)
    if os.path.isdir(subdir_path):
        # add a task setting the runner onto each state
        # we use the -s flag with the subdir name,
        # since experiment directories are structured with USPS state codes as directory names
        client.add_task(
            job_id=JOB_ID,
            docker_cmd="python %s -s %s" % (args.runner, subdir),
        )
client.monitor_job(job_id=JOB_ID)
