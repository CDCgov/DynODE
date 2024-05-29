"""
This Python script, when run, launches a Shiny app on http://localhost:8000/ that aids in visualizing
individual particles of posteriors produced by an experiment.
"""

import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import seaborn as sn
from shiny import App, render, ui, Session, reactive
from shinywidgets import output_widget, render_widget  
import os
import sys
import plotly.express as px
import pandas as pd
import numpy as np
import json
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


def get_azure_files(exp: str, jobid: str, state: str, scenario: str, azure_client: AzureClient) -> list[str]:
    """
    Reads in all files from the output blob `exp/jobid/state` in `azure_client.out_cont_client` 
    and stores them in the `shiny_cache` directory. 
    If the cache path already exists for that run, nothing is downloaded.

    Raises ValueError error if `exp/jobid/state/` does not exist in the output blob
    """
    
    # will override files in cache
    shiny_cache_path = "shiny_visualizers\\shiny_cache"
    azure_blob_path = os.path.join(exp, jobid, state)
    if scenario != "N/A": # if visualizing a scenario append to the path
        azure_blob_path = os.path.join(azure_blob_path, scenario)
    azure_blob_path = azure_blob_path.replace('\\', "/") + "/"

    download_directory(azure_client.out_cont_client, azure_blob_path, shiny_cache_path)
    return os.path.join(shiny_cache_path, azure_blob_path)
    
def visualize_vax_rates(cache_path):
    vax_rate = np.array(d["VACCINATION_RATES"])
    shp = vax_rate.shape
    age = ["0-17", "18-49", "50-64", "65+"]
    vax_dose = ["0 -> 1", "1 -> 2", "2 -> 3", "3 -> 3"]
    days = pd.Series(np.arange(shp[0])).repeat(shp[1] * shp[2]).to_list()
    ages = pd.Series(age * shp[0]).repeat(shp[2]).to_list()
    vax_doses = vax_dose * (shp[0] * shp[1])

    df = pd.DataFrame({
        "day": days,
        "age": ages,
        "vax_dose": vax_doses,
        "vax_rate": vax_rate.flatten()
    })
    fig = px.line(df, x="day", y="vax_rate", color="age", facet_col="vax_dose",
              facet_col_wrap=2, template="plotly_white")
    fig.update_traces(mode="lines", hovertemplate=None)
    fig.update_layout(hovermode="x")
    fig.show()
    
#shiny_visualizers\shiny_cache\projections\test_3\IL\noBoo_highIE\checkpoint_noBoo_highIE_0_214.json
def visualize_immunity(cache_path):
    f = open(os.path.join(cache_path, "checkpoint_noBoo_highIE_1_214.json").replace("\\", "/"), "r")
    d = json.load(f)
    age = ["0-17", "18-49", "50-64", "65+"]
    strain = ["XBB1", "XBB2", "JN1", "W", "X", "Y", "Z"]
    vax_rate = np.array(d["VACCINATION_RATES"])
    df = pd.DataFrame({
        'days': np.arange(365),
        'vax_rate': [v for v in vax_rate]
    })
    immunity = np.array(d["IMMUNITY_STRAIN"]).transpose((1, 2, 0))
    shp = immunity.shape

    days = pd.Series(np.arange(shp[0])).repeat(shp[1] * shp[2]).to_list()
    ages = pd.Series(age * shp[0]).repeat(shp[2]).to_list()
    strains = strain * (shp[0] * shp[1])

    df = pd.DataFrame({
        "day": days,
        "age": ages,
        "strain": strains,
        "immunity": immunity.flatten()
    })
    fig = px.line(df, x="day", y="immunity", color="age", facet_col="strain", facet_col_wrap=4, template="plotly_white")
    fig.update_yaxes(range=[0, 1])
    return fig


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
default_scenario_list=["N/A"]
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
            ui.input_selectize(
                "scenario",
                "Scenario",
                default_scenario_list,
                selected=default_scenario_list[0],
            ),
            ui.input_action_button("action_button", "Visualize Output"),  
            ui.output_text("counter"),
        ),
        ui.panel_main(ui.output_plot("plot", height="750px")),
    ),
)

def server(input, output, session: Session):
    @output
    @render.plot
    def plot(fig=None):
        experiment = input.experiment()
        tree_exp = tree_root.children[experiment]
        new_job_id_selections = [node.name for node in tree_exp.children.values()]
        # update the jobs able to be picked based on the currently selected experiment
        selected_jobid = input.job_id() if input.job_id() in new_job_id_selections else new_job_id_selections[0]
        ui.update_selectize("job_id", choices=new_job_id_selections, selected=selected_jobid)
        tree_job = tree_exp.children[selected_jobid]
        new_state_selections = [node.name for node in tree_job.children.values()]
        selected_state = input.state() if input.state() in new_state_selections else new_state_selections[0]
        # update the states able to be picked based on the currently selected job
        ui.update_selectize("state", choices=[node.name for node in tree_job.children.values()], selected=selected_state)
        tree_state = tree_job.children[selected_state]
        if list(tree_state.children.values())[0].children:
            new_scenario_selections = [node.name for node in tree_state.children.values()]
            selected_scenario = input.scenario() if input.scenario() in new_scenario_selections else new_scenario_selections[0]
            ui.update_selectize("scenario", choices=new_scenario_selections, selected=selected_scenario)
        return fig
    @render_widget
    @reactive.event(input.action_button)
    def counter():
        print("running")
        exp = input.experiment()
        job_id = input.job_id()
        state = input.state()
        scenario = input.scenario()
        cache_path = get_azure_files(exp, job_id, state, scenario, azure_client)
        fig = visualize_immunity(cache_path)
        plot(fig=fig)
        return fig

app = App(app_ui, server)
app.run()
