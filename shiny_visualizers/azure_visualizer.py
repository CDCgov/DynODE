"""
This Python script, when run, launches a Shiny app on http://localhost:8000/ that aids in visualizing
individual particles of posteriors produced by an experiment.
"""

import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import seaborn as sn
from shiny import App, render, ui
# from shiny import reactive
# from shiny.express import input, ui, render, expressify 
# import utils
import os
import sys
sys.path.append("..")
from mechanistic_model.covid_initializer import CovidInitializer
from mechanistic_model.mechanistic_inferer import MechanisticInferer
from cfa_azure.clients import AzureClient
INITIALIZER_USED = CovidInitializer
INFERER_USED = MechanisticInferer
# INPUT_BLOB_NAME = "scenarios-test-container"
OUTPUT_BLOB_NAME = "example-output-scenarios-mechanistic"
print("Connecting to Azure Storage")
client = AzureClient(config_path="secrets/configuration_cfaazurebatchprd.toml")
# client.set_input_container(INPUT_BLOB_NAME, "input")
client.set_output_container(OUTPUT_BLOB_NAME, "output")
container_client = client.out_cont_client
print("Retrieving and Organizing Azure File Structure (approx 10 seconds)")
blobs = container_client.list_blob_names()
class Node:
    def __init__(self, name):
        self.name = name
        self.children = {}
 
def construct_tree(file_paths):
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
  
tree_root = construct_tree(blobs)
 
# print(tree_root.children)
experiment_names = [node.name for node in tree_root.children.values()]
# these are used to initally populate the selectors and are then dynamically updated based on the experiment chosen
default_job_list = [node.name for node in tree_root.children[experiment_names[0]].children.values()]
default_state_list = [node.name for node in tree_root.children[experiment_names[0]].children[default_job_list[0]].children.values()]
app_ui = ui.page_fixed(
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
            )
        ),
        ui.panel_main(ui.output_plot("plot", height="750px")),
    ),
)
    
def server(input, output, session):
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
        download_run
        return fig

app = App(app_ui, server)
app.run()
