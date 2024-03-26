"""
a script which is passed two flags, one to dictate the experiment folder on which to run
and another to specify the location of the state-specific runner script, which will perform analysis on a single state.
"""

import argparse
import os
import subprocess
from cfa_azure.clients import AzureClient
import cfa_azure.helpers as helpers
import os


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
    help="Path to single state runner Python file",
    required=True,
)
args = parser.parse_args()
client = AzureClient(config_path="secrets/configuration_cfaazurebatchprd.toml")
# client = AzureClient(config_path=None)
# turn debugging on, this is required
# client.set_debugging(True)

client.package_and_upload_dockerfile(
    registry_name="cfaprdbatchcr", repo_name="scenarios", tag="expinfer2"
)

# create the input and output blobs
client.set_input_container("scenarios-test-container", "input")
client.set_output_container("example-output-scenarios-mechanistic", "output")

# set the scaling of the pool:autoscale
client.set_scaling(
    mode="fixed", dedicated_nodes=2, timeout=150
)
client.create_pool(pool_name="scenarios_2_node")
# or use a certain pool
# client.use_pool(pool_name="scenarios_2_node")
in_files = helpers.list_files_in_container(
                client.input_container_name, client.sp_credential, client.config
            ) 
job_id = "scenarios_inference_run_4"
# command to run the job
client.add_job(job_id=job_id, input_files=in_files)
for subdir in os.listdir(args.folder):
        subdir_path = os.path.join(args.folder, subdir)
        if os.path.isdir(subdir_path):
            # add a task setting the runner onto each state
            print("python %s -s %s"%(args.runner, subdir))
            client.add_task(job_id=job_id, task_id = job_id + subdir, docker_cmd="python %s -s %s"%(args.runner, subdir))
client.monitor_job(job_id=job_id)
