"""
This Python script, when run, launches a Shiny app on http://localhost:8000/ that aids in visualizing
individual particles of posteriors produced by an experiment.
"""

import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import seaborn as sn
from shiny import App, render, ui, Session, reactive
import os
import sys
sys.path.append("c:\\Users\\uva5\\Documents\\GitHub\\cfa-scenarios-model")
from mechanistic_model.covid_initializer import CovidInitializer
from mechanistic_model.mechanistic_inferer import MechanisticInferer
import utils
from cfa_azure.clients import AzureClient
from cfa_azure.helpers import download_directory
INITIALIZER_USED = CovidInitializer
INFERER_USED = MechanisticInferer
INPUT_BLOB_NAME = "scenarios-test-container"
OUTPUT_BLOB_NAME = "example-output-scenarios-mechanistic"

def build_azure_connection(config_path = "secrets/configuration_cfaazurebatchprd.toml"):    
    client = AzureClient(config_path=config_path)
    client.set_input_container(INPUT_BLOB_NAME, "input")
    client.set_output_container(OUTPUT_BLOB_NAME, "output")
    return client

def get_blob_names(azure_client: AzureClient):
    return azure_client.out_cont_client.list_blob_names()

class Node:
    # a helper class to store directories in a tree
    def __init__(self, name):
        self.name = name
        self.children = {}
 
def construct_tree(file_paths):
    """
    given a list of directories from get_blob_names() returns a tree of each folder and its subdirectories
    leaf nodes are files with a "." like .txt or .json
    """
    root = Node("/")
    for path in file_paths:
        if "." in path: # indicates this is an actual file like .txt or .json or .out
            directories, filename = path.rsplit("/", 1)
            current_node = root
            for directory in directories.split("/"):
                if directory not in current_node.children:
                    current_node.children[directory] = Node(directory)
                current_node = current_node.children[directory]
            current_node.children[filename] = Node(filename)
    return root


def get_azure_files(exp: str, jobid: str, state: str, azure_client: AzureClient) -> list[str]:
    """
    Reads in all files from the output blob `exp/jobid/state` in `azure_client.out_cont_client` 
    and stores them in the `shiny_cache` directory. 
    If the cache path already exists for that run, nothing is downloaded.

    Raises ValueError error if `exp/jobid/state/` does not exist in the output blob
    """
    
    # will override files in cache
    shiny_cache_path = "shiny_visualizers\\shiny_cache"
    azure_blob_path = os.path.join(exp, jobid, state).replace('\\', "/") + "/"
    download_directory(azure_client.out_cont_client, azure_blob_path, shiny_cache_path)
    


print("Connecting to Azure Storage")
azure_client = build_azure_connection()
print("Retrieving and Organizing Azure File Structure (approx 10 seconds)")
blobs = get_blob_names(azure_client)
tree_root = construct_tree(blobs)
 
# now that we have all the paths stored in a Tree, all these operations are quick
# these are used to initally populate the selectors 
# and are then dynamically updated based on the experiment chosen
experiment_names = [node.name for node in tree_root.children.values()]
default_job_list = [node.name for node in tree_root.children[experiment_names[0]].children.values()]
default_state_list = [node.name for node in tree_root.children[experiment_names[0]].children[default_job_list[0]].children.values()]
####################################### SHINY CODE ################################################
app_ui = ui.page_fluid(
    ui.h2("Visualizing Immune History"),
    ui.markdown(
        """
    """
    ),
    ui.layout_sidebar(
        ui.panel_sidebar(
            ui.input_selectize(
                "experiment",
                "Experiment Name",
                experiment_names,
                multiple=False,
                selected=experiment_names[0],
            ),
            ui.input_selectize(
                "job_id",
                "Job ID",
                default_job_list,
                multiple=False,
                selected=default_job_list[0],
            ),
            ui.input_selectize(
                "state",
                "State",
                default_state_list,
                selected=default_state_list[0],
            ),
            ui.input_action_button("action_button", "Download Output"),  
            ui.output_text("counter"),
        ),
        ui.panel_main(ui.output_plot("plot", height="750px")),
    ),
)

def server(input, output, session: Session):
    @output
    @render.plot
    def plot():
        experiment = input.experiment()
        new_job_id_selections = [node.name for node in tree_root.children[experiment].children.values()]
        # update the jobs able to be picked based on the currently selected experiment
        selected_jobid = input.job_id() if input.job_id() in new_job_id_selections else new_job_id_selections[0]
        ui.update_selectize("job_id", choices=new_job_id_selections, selected=selected_jobid)
        new_state_selections = [node.name for node in tree_root.children[experiment].children[selected_jobid].children.values()]
        selected_state = input.state() if input.state() in new_state_selections else new_state_selections[0]
        # update the states able to be picked based on the currently selected job
        ui.update_selectize("state", choices=[node.name for node in tree_root.children[experiment].children[selected_jobid].children.values()], selected=selected_state)
        fig, axs = plt.subplots(2, 1)
        return fig
    @render.text()
    @reactive.event(input.action_button)
    def counter():
        print("running")
        exp = input.experiment()
        job_id = input.job_id()
        state = input.state()
        if os.path.exists(os.path.join("shiny_cache", exp, job_id, state)):
            print("path exists")
        else:
            print("downloading")
            requested_files = [node.name for node in tree_root.children[exp].children[job_id].children[state].children.values()]
            print(requested_files)
            file_names = get_azure_files(exp, job_id, state, azure_client, requested_files)
            print(file_names)
        return str(file_names)

app = App(app_ui, server)
app.run()
