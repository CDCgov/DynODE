import os

import cfa_azure.helpers
from azure.core.paging import ItemPaged
from cfa_azure.clients import AzureClient


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
        self._container_registry_name = container_registry_name
        self._image_repo_name = image_repo_name
        self._docker_image_name = docker_image_name
        self.job_id = job_id
        self.job_launched = False
        self.set_all_paths(experiment_directory, experiment_name)
        self._setup_azure_client(path_to_dockerfile)

    def set_experiment_paths(
        self, experiments_folder_name, experiment_name
    ) -> None:
        """
        identifies the location of the experiment on the local machine, docker enviorment, and storage blob input data
        NOTE: we prepend /input/ to all docker paths since our azure blob is MOUNTED onto a docker container

        Parameters
        ----------
        experiments_folder_name : str
            name of the top level folder holding individual experiments
        experiment_name : str
            name of the experiment being run, a subdirectory of `experiments_folder_name`
        """
        self._experiments_folder_name = experiments_folder_name
        self._experiment_name = experiment_name
        # using the experiment name, get the local (this machine) and docker (azure batch node) paths to each file
        # NOTE: we prepend /input/ to the docker path since our azure blob is MOUNTED onto a docker container
        # and the mount appears as a folder /input
        experiment_path_local = os.path.join(
            experiments_folder_name, experiment_name
        )
        experiment_path_docker = os.path.join(
            "/input/", experiments_folder_name, experiment_name, self.job_id
        )
        experiment_path_blob = os.path.join(
            experiments_folder_name, experiment_name, self.job_id
        )
        assert os.path.exists(experiment_path_local), (
            "Experiment does not exist at expected directory %s"
            % experiment_path_local
        )
        self.experiment_path_local = experiment_path_local
        self.experiment_path_docker = experiment_path_docker
        self.experiment_path_blob = experiment_path_blob

    def set_runner_paths(
        self, runner_script_filename: str = "run_task.py"
    ) -> None:
        """identifies the local and docker path of the runner script, on which each state is run.
        NOTE: runner scripts MUST be inside the top level of the experiment, e.g. exp/experiment_name/run_task.py

        Parameters
        ----------
        runner_script_filename : str, optional
            filename of the runner script, by default "run_task.py"
        """
        # get a path to the run_task python script, both on this machine and in the docker blob
        runner_path_local = os.path.join(
            self.experiment_path_local, runner_script_filename
        )
        runner_path_docker = os.path.join(
            self.experiment_path_docker, runner_script_filename
        )
        assert os.path.exists(runner_path_local), (
            "make sure your %s is found inside of your experiment folder, was not found at %s"
            % (runner_script_filename, runner_path_local)
        )
        self.runner_path_local = runner_path_local
        self.runner_path_docker = runner_path_docker

    def set_state_paths(self, states_folder_name: str = "states") -> None:
        """identifies the location of the directory containing all states run, both locally and within the docker enviorment

        Parameters
        ----------
        states_folder_name : str, optional
            folder containing the states to be run, by default "states"
        """
        states_path_local = os.path.join(
            self.experiment_path_local, states_folder_name
        )
        states_path_docker = os.path.join(
            self.experiment_path_docker, states_folder_name
        )
        # check that the files are in the right place locally
        assert os.path.exists(states_path_local), (
            "was unable to find which states you are launching within %s"
            % states_path_local
        )
        self.states_path_local = states_path_local
        self.states_path_docker = states_path_docker

    def set_all_paths(
        self,
        experiments_folder_name: str,
        experiment_name: str,
        runner_script_filename: str = "run_task.py",
        states_folder_name: str = "states",
    ):
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
        runner_script_filename : str, optional
            filename of the runner script, by default "run_task.py"
        states_folder_name : str, optional
            folder containing the states to be run, by default "states"

        """
        self.set_experiment_paths(experiments_folder_name, experiment_name)
        self.set_runner_paths(runner_script_filename)
        self.set_state_paths(states_folder_name)

    def _setup_azure_client(self, path_to_dockerfile="./Dockerfile"):
        """sets up an AzureClient with cfa_azure package,
        authenticates and uploads experiment and docker image
        to storage blob and container registry respectively

        Parameters
        ----------
        path_to_dockerfile : str, optional
            path to the Dockerfile, by default "./Dockerfile"
        """
        # start up azure client with authentication toml
        azure_client = AzureClient(config_path=self.azure_config_toml)
        # run `docker build` using the Dockerfile in the cwd, apply tag
        azure_client.package_and_upload_dockerfile(
            path_to_dockerfile=path_to_dockerfile,
            registry_name=self._container_registry_name,
            repo_name=self._image_repo_name,
            tag=self._docker_image_name,
        )
        # create the input and output blobs, for now they must be named /input and /output
        azure_client.set_input_container(
            "scenarios-mechanistic-input", "input"
        )
        azure_client.set_output_container(
            "scenarios-mechanistic-output", "output"
        )

        self.azure_client = azure_client

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

    def _upload_experiment_to_blob(self):
        """
        upload the experiment folder so that the runner_path_docker & states_path_docker point to the correct places
        here we pass `location=self.experiment_path_blob` because we are not uploading from the docker container
        therefore we dont need the /input/ mount directory
        """
        self.azure_client.upload_files_in_folder(
            [self.experiment_path_local],
            "scenarios-mechanistic-input",
            location_in_blob=self.experiment_path_blob,
        )

    def launch_states(self, depend_on_task_ids: list[str] = None) -> list[str]:
        """Launches an Azure Batch job under `self.job_id`,
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
                    depends_on=depend_on_task_ids,
                    name_suffix=statedir,
                )
                # append this list onto our running list of tasks
                task_ids += task_id
        return task_ids

    def _find_postprocess_file_docker(
        self, postprocess_folder_name, postprocess_filename
    ) -> str:
        """given a postprocess script filename validates its existance
        in the local experiment, returns the file's path on the docker
        environment

        Parameters
        ----------
        postprocessing_folder_name: str
            name of the folder in which postprocessing scripts are held,
        postprocess_filename : str
            filename of the postprocess script. Able to
            correctly parse paths as well as raw filenames.

        Returns
        -------
        str
            path to the postprocess script as executed on the docker machine
            (they are uploaded to the input mount)

        Raises
        ------
        FileNotFoundError
            if `postprocess_filename` does not exist in the experiment directory
            nor the `postprocess_folder_name`, raises.
        """
        if os.path.exists(
            os.path.join(self.experiment_path_local, postprocess_filename)
        ):
            postprocess_docker_path = os.path.join(
                self.experiment_path_docker, postprocess_filename
            )
        # if user does not prepend the folder name catch that error
        elif os.path.exists(
            os.path.join(
                self.experiment_path_local,
                postprocess_folder_name,
                postprocess_filename,
            )
        ):
            postprocess_docker_path = os.path.join(
                self.experiment_path_docker,
                postprocess_folder_name,
                postprocess_filename,
            )
        else:
            raise FileNotFoundError(
                "Unable to find postprocessing script %s in your "
                "experiment folder, make sure you are specifying "
                "paths relative to the experiment folder"
                % postprocess_filename
            )
        return postprocess_docker_path

    def launch_postprocess(
        self,
        execution_order: list[str | list[str]],
        depend_on_task_ids: list[str],
        postprocess_folder_name: str = "postprocessing_scripts",
    ) -> list[str]:
        """Launches postprocessing scripts identified by `execution_order`
        in the order they are passed in the list. List elements are
        treated as bundles of postprocess scripts that are able to be run
        simultaneously. Script filenames given to `execution_order` must
        be found in either

        Example
        ----------
        ```
        main_tasks = self.launch_states()
        #launch 3 postprocess tasks to execute simultaneously
        ex_order = [["1.py", "2.py", "3.py"]]
        #launch 3 postprocess tasks to execute sequentially (note no list nesting)
        ex_order = ["1.py", "2.py", "3.py"]
        ex_order = [["1.py",] ["2.py"], ["3.py"]]
        #launch 3 postprocess tasks, 1 and 2 to execute together, 3 last
        ex_order = [("1.py", "2.py"), "3.py"]
        postprocess = self.launch_postprocess(execution_order = ex_order, depend_on_task_ids = main_tasks)
        ```
        Parameters
        ----------
        execution_order: list[str | list[str]]
            list of paths to each postprocess script, lists of scripts imply
            simultaneous execution while ordering of the top level list
            implies order of execution.
        depend_on_task_ids : list[str]
            list of task ids on which postprocessing scripts depend on finishing to start themselves
        postprocessing_folder_name: str, optional
            name of the folder in which postprocessing scripts are held,
            by default "postprocessing_scripts/"

        Returns
        -------
        list[str]
            list of each postprocess task_id launched
        """
        # postprocess_task_ids added one execution bundle at a time
        # so within-bundles run simultaneously but bundles run sequentially
        postprocess_task_ids = []
        for execution in execution_order:
            # convert str to list[str] even if only single element
            execution_bundle: list[str] = (
                [execution] if isinstance(execution, str) else execution
            )
            execution_bundle_ids = []
            for postprocess_filename in execution_bundle:
                # translate paths to docker paths
                # may raise FileNotFoundError if unable to locate ppfile
                postprocess_docker_path = self._find_postprocess_file_docker(
                    postprocess_folder_name, postprocess_filename
                )
                # postprocess_task_ids contains all previous execution_bundles
                # task_ids, but not any task_ids from the current bundle
                task_id = self.azure_client.add_task(
                    job_id=self.job_id,
                    docker_cmd="python %s -j %s"
                    % (postprocess_docker_path, self.job_id),
                    depends_on=depend_on_task_ids + postprocess_task_ids,
                )
                execution_bundle_ids += task_id
            # bundle completed, add those task ids to the running list
            postprocess_task_ids += execution_bundle_ids

        return postprocess_task_ids


def build_azure_connection(
    config_path: str = "secrets/configuration_cfaazurebatchprd.toml",
    input_blob_name: str = "scenarios-mechanistic-input",
    output_blob_name: str = "scenarios-mechanistic-output",
) -> AzureClient:
    """builds an AzureClient and connects input and output blobs
    found in INPUT_BLOB_NAME and OUTPUT_BLOB_NAME global vars.

    Parameters
    ----------
    config_path : str, optional
        path to your authentication toml, should not be public, by default "secrets/configuration_cfaazurebatchprd.toml"

    Returns
    -------
    AzureClient object
    """
    client = AzureClient(config_path=config_path)
    client.set_input_container(input_blob_name, "input")
    client.set_output_container(output_blob_name, "output")
    return client


def get_blob_names(
    azure_client: AzureClient, name_starts_with: str
) -> ItemPaged[str]:
    """returns the blobs stored in OUTPUT_BLOB_NAME on Azure Storage Account

    Parameters
    ----------
    azure_client : AzureClient
        the Azure client with the correct authentications to access the data, built from `build_azure_connection`

    Returns
    -------
    Iterator[str]
        an iterator of each blob, including directories: e.g\n
        root \n
        root/fol1 \n
        root/fol1/fol2 \n
        root/fol1/fol2/file.txt \n
    """
    return azure_client.out_cont_client.list_blob_names(
        name_starts_with=name_starts_with
    )


def download_directory_from_azure(
    azure_client: AzureClient,
    azure_dirs: list[str] | str,
    dest: str,
    overwrite: bool = False,
) -> list[str]:
    """Downloads one or multiple azure directories, including all subdirectories
    found within azure_dir. Preserves their directory structure, appending them to `dest`.

    Parameters
    ----------
    azure_client : AzureClient
        AzureClient authenticated and connected to storage from which this action will be preformed
    azure_dir : list[str] | str
        azure directory/directories to download
    dest : str
        location to place downloaded `azure_dirs`
    override : bool, optional
        whether to overwrite directories matching `azure_dirs` within `dest`, by default False

    Returns
    -------
    list[str]
        directories that were successfully written

    Raises
    --------
    ValueError if a directory within `azure_dirs` does not exist on storage.
    """
    written_dirs = []
    if isinstance(azure_dirs, str):
        azure_dirs = [azure_dirs]
    for azure_dir in azure_dirs:
        dest_path = os.path.join(dest, azure_dir)
        # if path does not exist on local OR we are overwriting anyways
        if not os.path.exists(dest_path) or overwrite:
            try:
                cfa_azure.helpers.download_directory(
                    container_name=azure_client.output_container_name,
                    blob_service_client=azure_client.blob_service_client,
                    dest_path=dest,
                    src_path=azure_dir,
                )
            except ValueError as e:
                raise ValueError(
                    "failed to download %s, but not before successfully downloading %s"
                    % (dest_path, str(written_dirs))
                ) from e
        written_dirs.append(dest_path)
    return written_dirs
