from cfa_azure.clients import AzureClient
import cfa_azure.helpers as helpers
import os

client = AzureClient(config_path="secrets/configuration_cfaazurebatchprd.toml")
# client = AzureClient(config_path=None)
# turn debugging on, this is required
# client.set_debugging(True)

client.package_and_upload_dockerfile(
    registry_name="cfaprdbatchcr", repo_name="scenarios-test", tag="test3"
)

# create the input and output blobs
client.set_input_container("scenarios-test-container", "input")
client.set_output_container("example-output-scenarios-mechanistic", "output")

# set the scaling of the pool:autoscale
client.set_scaling(
    mode="fixed", dedicated_nodes=2#, autoscale_formula_path="./autoscale_formula.txt"
)
# if fixed mode is desired, do the following:
# client.set_scaling(mode="fixed")

# create the pool
# client.create_pool(pool_name="scenarios_2_node")
# or use a certain pool
client.use_pool(pool_name="scenarios_2_node")

# upload files
# client.upload_files_in_folder(["data"])
# in_files = []
# for folder, _, file in os.walk(os.path.realpath("./data")):
#         for file_name in file:
#             in_files.append(file_name)
print("MYPRINTS")
print(client.input_container_name)
print(client.config)
print(client.full_container_name)
print(client.container_image_name)
in_files = helpers.list_files_in_container(
                client.input_container_name, client.sp_credential, client.config
            ) 
job_id = "scenarios_test_run_20"
# command to run the job
client.add_job(job_id=job_id, input_files=in_files)
# docker_cmd = "python ./example_end_to_end_run.py -state "
# for state in states:
#     docker_cmd = "python ./example_end_to_end_run.py -state " + str(state)
client.add_task(job_id=job_id, docker_cmd="python /app/experiment_runner.py --folder /app/exp/three_state_experiment -- runner /app/exp/three_state_experiment/single_state_runner")
client.monitor_job(job_id=job_id)

# close down the jobs, required when using debug is True
# look into https://learn.microsoft.com/en-us/python/api/azure-batch/azure.batch.operations.joboperations?view=azure-python#azure-batch-operations-joboperations terminate
# client.delete_job(job_id="runtest6")