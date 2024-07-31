from cfa_azure.clients import AzureClient
from mechanistic_model.utils import find_files, sort_filenames_by_suffix
import os


class AzureExperimentLauncher:
    def __init__(
        self,
        experiment_name,
        job_id: str,
        azure_config_toml: str = "secrets/configuration_cfaazurebatchprd.toml",
        experiment_directory: str = "exp",
        container_registry_name: str = "cfaprdbatchcr",
        image_repo_name: str = "scenarios",
        docker_image_name: str = "scenarios-image",
        path_to_dockerfile: str = "./Dockerfile",
    ):
        """Creates an Azure Experiment Launcher, identifying experiment, states, and runner script paths.
        Then attempts to authenticate with azure using `azure_config_toml` and upload
        experiment and docker image to Azure blob storage and container repo respectively.

        Parameters
        ----------
        experiment_name : str
            name of the experiment being run (should match the folder name)
        job_id : str
            job id of the job being launched, must be unique
        azure_config_toml : str, optional
            path to azure configuration toml, by default "secrets/configuration_cfaazurebatchprd.toml"
        experiment_directory : str, optional
            path to directory containing all experiments, by default "exp"
        container_registry_name : str, optional
            container registry to which docker image will be uploaded, by default "cfaprdbatchcr"
        image_repo_name : str, optional
            container repository to which docker image will be uploaded, by default "scenarios"
        docker_image_name : str, optional
            tag given to uploaded docker image, by default "scenarios-image"
        path_to_dockerfile : str, optional
            path to the `Dockerfile` used to build docker image, by default "./Dockerfile"
        """
        self.azure_config_toml = azure_config_toml
        self.experiment_directory = experiment_directory
        self.experiment_name = experiment_name
        self.container_registry_name = container_registry_name
        self.image_repo_name = image_repo_name
        self.docker_image_name = docker_image_name
        self.job_id = job_id
        (
            self.experiment_path_local,
            self.experiment_path_docker,
            self.experiment_path_blob,
            self.states_path_local,
            self.states_path_docker,
            self.runner_path_local,
            self.runner_path_docker,
        ) = self._identify_local_and_docker_paths(
            experiment_directory, experiment_name
        )
        self.azure_client = self._setup_azure_client(path_to_dockerfile)

    def _identify_local_and_docker_paths(
        self, experiments_folder_name: str, experiment_name: str, job_id: str
    ) -> tuple[str, str, str, str, str, str]:
        """Using the name of the experiment being run, along with where experiments are stored
        identify and validate that necessary files and folders exist on the users local machine.
        Then deduce the locations of those same files on Azure Storage once they are uploaded.

        NOTE: we prepend /input/ to all docker paths since our azure blob is MOUNTED onto a docker container

        Parameters
        ----------
        experiments_folder_name : str
            name of the top level folder holding individual experiments
        experiment_name : str
            name of the experiment being run, a subdirectory of `experiments_folder_name`
        job_id : str
            job id of the job being launched, must be unique

        Returns
        ----------
        tuple[str] local/docker paths to relevant files
        `experiment_path_local`: path to the experiment folder (experiments_folder_name + experiment_name) \n
        `experiment_path_docker`: path to the experiment folder on the docker machine \n
        `experiment_path_blob`: path to the experiment folder within the storage blob \n
        `states_path_local`: path to the states folder within the experiment\n
        `states_path_local`: path to the state folder within the experiment on the docker machine\n
        `runner_path_local`: path to the run_task script to run individual states\n
        `runner_path_docker`: path to the run_task script on the docker machine\n

        """
        # using the experiment name, get the local (this machine) and docker (azure batch node) paths to each file
        # NOTE: we prepend /input/ to the docker path since our azure blob is MOUNTED onto a docker container
        # and the mount appears as a folder /input
        experiment_path_local = os.path.join(
            experiments_folder_name, experiment_name
        )
        experiment_path_docker = os.path.join(
            "/input/", experiments_folder_name, experiment_name, job_id
        )
        experiment_path_blob = os.path.join(
            experiments_folder_name, experiment_name, job_id
        )
        # get a path to the run_task python script, both on this machine and in the docker blob
        runner_path_local = os.path.join(experiment_path_local, "run_task.py")
        runner_path_docker = os.path.join(
            experiment_path_docker, "run_task.py"
        )
        # get a path to the `states` folder, both on this machine and in the docker blob
        states_path_local = os.path.join(experiment_path_local, "states")
        states_path_docker = os.path.join(experiment_path_docker, "states")
        # check that the files are in the right place locally
        assert os.path.exists(
            runner_path_local
        ), "make sure your run_task.py is found inside of your experiment folder"
        assert os.path.exists(
            states_path_local
        ), "please rerun this file specifying the experiment directory, "
        "it must include a /states/ folder within it containing the state folders filled with configs"
        return (
            experiment_path_local,
            experiment_path_docker,
            experiment_path_blob,
            states_path_local,
            states_path_docker,
            runner_path_local,
            runner_path_docker,
        )

    def _setup_azure_client(self, path_to_dockerfile="./Dockerfile"):
        """sets up an AzureClient with cfa_azure package,
        authenticates and uploads experiment and docker image
        to storage blob and container registry respectively

        Parameters
        ----------
        path_to_dockerfile : str, optional
            path to the Dockerfile, by default "./Dockerfile"

        Returns
        ----------
        AzureClient a cfa_azure client after packaging and uploading the docker image and experiment files
        """
        # start up azure client with authentication toml
        azure_client = AzureClient(config_path=self.azure_config_toml)
        # run `docker build` using the Dockerfile in the cwd, apply tag
        azure_client.package_and_upload_dockerfile(
            path_to_dockerfile=path_to_dockerfile,
            registry_name=self.container_registry_name,
            repo_name=self.image_repo_name,
            tag=self.docker_image_name,
        )
        # create the input and output blobs, for now they must be named /input and /output
        azure_client.set_input_container(
            "scenarios-mechanistic-input", "input"
        )
        azure_client.set_output_container(
            "scenarios-mechanistic-output", "output"
        )

        # upload the experiment folder so that the runner_path_docker & states_path_docker point to the correct places
        # here we pass `location=self.experiment_path_blob` because we are not uploading from the docker container
        # therefore we dont need the /input/ mount directory
        azure_client.upload_files_in_folder(
            [self.experiment_path_local],
            "scenarios-mechanistic-input",
            location=self.experiment_path_blob,
        )
        return azure_client

    def set_resource_pool(
        self,
        pool_name: str = "scenarios_4cpu_pool",
        create: bool = False,
        timeout_mins: int = 5,
        autoscale_formula_path: str = "secrets/autoscale.txt",
    ):
        """Sets or creates the Azure resource pool, defining the type of compute spun up
        once jobs are run on `self.azure_client`.

        Parameters
        ----------
        pool_name : str, optional
            name of the pool to create/connect to, if create is False, this pool must already exist, by default "scenarios_4cpu_pool"
        create : bool, optional
            whether or not to create `pool_name`, defaults to autoscale formula within `secrets/autoscale.txt`,
            will raise Errors if pool with that name already exists, by default False
        timeout_mins : int
            if creating a new pool, set the timeout window where idle nodes will leave the pool
        autoscale_formula_path : str
            path to autoscale formula used as logic to scale up or down the resource pool on job launch
        """
        if create:
            self.azure_client.set_pool_info(
                mode="autoscale",
                autoscale_formula_path=autoscale_formula_path,
                timeout=timeout_mins,
            )
            self.azure_client.create_pool(pool_name=pool_name)
        else:
            self.azure_client.set_pool(pool_name)

    def launch_states(
        self,
    ) -> list[str]:
        """Launches an Azure Batch job under `self.job_id`,
        populating it with tasks for each subdirectory within your experiment's `states` directory
        passing each state name to `run_task.py` with the -s flag and the job_id with the -j flag.

        Returns
        -------
        list[str]
            list of all tasks launched under `job_id`
        """
        # command to run the job
        self.azure_client.add_job(job_id=self.job_id)
        task_ids = []
        # add a task for each state directory in the states folder of this experiment
        for statedir in os.listdir(self.states_path_local):
            statedir_path = os.path.join(self.states_path_local, statedir)
            if os.path.isdir(statedir_path):
                # add a task setting the runner onto each state
                # we use the -s flag with the subdir name,
                # since experiment directories are structured with USPS state codes as directory names
                # also include the -j flag to specify the jobid
                task_id = self.azure_client.add_task(
                    job_id=self.job_id,
                    docker_cmd="python %s -s %s -j %s"
                    % (self.runner_path_docker, statedir, self.job_id),
                )
                # append this list onto our running list of tasks
                task_ids += task_id
        return task_ids

    def launch_postprocess(
        self,
        depend_on_task_ids: str,
    ) -> list[str]:
        """Launches postprocessing scripts from within `experiment_path_local` identified by
        the postprocess_states tag and an optional suffix.
        All postprocessing scripts depend on completion of tasks within `depend_on_task_ids`
        along with any postprocessing scripts with lower suffix.
        Eg. `postprocess_states_2.py` depends on `postprocess_states_1.py` to finish first.

        Parameters
        ----------
        depend_on_task_ids : list[str]
            list of task ids on which postprocessing scripts depend on finishing to start themselves

        Returns
        -------
        list[str]
            list of each postprocess task_id launched
        """
        # get all paths to postprocess scripts on this machine
        postprocess_scripts = find_files(
            self.experiment_path_local, filename_contains="postprocess_states"
        )
        # lets sort postprocess scripts in order of their suffix
        postprocess_scripts = sort_filenames_by_suffix(postprocess_scripts)
        # [] means no postprocessing scripts in this experiment
        if postprocess_scripts:
            # translate paths to docker paths
            postprocess_scripts_docker = [
                os.path.join(self.experiment_path_docker, filename)
                for filename in postprocess_scripts
            ]
            postprocess_task_ids = []
            for postprocess_script in postprocess_scripts_docker:
                # depends_on flag requires postprocessing scripts to await completion of all previously run tasks
                # this means postprocess_states.py requires all states to finish
                # postprocessing scripts will require all earlier postprocessing scripts to finish before starting as well.
                postprocess_task_id = self.client.add_task(
                    job_id=self.job_id,
                    docker_cmd="python %s -j %s"
                    % (postprocess_script, self.job_id),
                    depends_on=depend_on_task_ids + postprocess_task_ids,
                )
                postprocess_task_ids += postprocess_task_id

        return postprocess_task_ids
