"""
This Python script, when run, launches a Shiny app on http://localhost:8000/ that aids in visualizing
individual particles of posteriors produced by an experiment.
"""

# ruff: noqa: E402
import os
import sys
from typing import Union

import numpy as np
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
from shiny import App, Session, reactive, ui
from shinywidgets import output_widget, render_widget

sys.path.append("c:\\Users\\uva5\\Documents\\GitHub\\cfa-scenarios-model")
from cfa_azure.clients import AzureClient
from cfa_azure.helpers import download_directory

from mechanistic_model.covid_initializer import CovidInitializer
from mechanistic_model.mechanistic_inferer import MechanisticInferer

INITIALIZER_USED = CovidInitializer
INFERER_USED = MechanisticInferer
INPUT_BLOB_NAME = "scenarios-mechanistic-input"
OUTPUT_BLOB_NAME = "scenarios-mechanistic-output"
# this will reduce the time it takes to load the azure connection, but only shows
# one experiment worth of data, which may be what you want...
#  leave empty ("") to explore all experiments
PRE_FILTER_EXPERIMENTS = "example_output"
# when loading the overview timelines csv for each run, columns
# are expected to have names corresponding to the type of plot they create
# vaccination_0_17 specifies the vaccination_ plot type, multiple columns may share
# a plot type, e.g: vaccination_0_17, vaccination_18_49,...
# it is assumed if they share a plot type they will appear on the same plot together
OVERVIEW_PLOT_TYPES = np.array(
    [
        "seasonality_coef",
        "vaccination_",
        "total_infection_incidence",
        "_strain_proportion",
        "_external_introductions",
        "_average_immunity",
    ]
)
# plot titles for each type of plot
OVERVIEW_PLOT_TITLES = np.array(
    [
        "Seasonality Coefficient",
        "Vaccination Rate By Age",
        "Total Infection Incidence",
        "Strain Proportion of New Infections",
        "External Introductions by Strain",
        "Average Population Immunity Against Strains",
    ]
)
OVERVIEW_SUBPLOT_HEIGHT = 150
OVERVIEW_SUBPLOT_WIDTH = 1000


def build_azure_connection(
    config_path="secrets/configuration_cfaazurebatchprd_new_sp.toml",
):
    """builds an AzureClient and connects input and output blobs
    found in INPUT_BLOB_NAME and OUTPUT_BLOB_NAME global vars.

    Parameters
    ----------
    config_path : str, optional
        path to your authentication toml, should not be public, by default "secrets/configuration_cfaazurebatchprd_new_sp.toml"

    Returns
    -------
    AzureClient object
    """
    client = AzureClient(config_path=config_path)
    client.set_input_container(INPUT_BLOB_NAME, "input")
    client.set_output_container(OUTPUT_BLOB_NAME, "output")
    return client


def get_blob_names(azure_client: AzureClient):
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
        name_starts_with=PRE_FILTER_EXPERIMENTS
    )


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
        # indicates this is an actual file like .txt or .json or .out
        if "." in path:
            directories, filename = path.rsplit("/", 1)
            current_node = root
            for directory in directories.split("/"):
                if directory not in current_node.children:
                    current_node.children[directory] = Node(directory)
                current_node = current_node.children[directory]
            current_node.children[filename] = Node(filename)
    return root


def get_azure_files(
    exp: str, jobid: str, state: str, scenario: str, azure_client: AzureClient
) -> list[str]:
    """
    Reads in all files from the output blob `exp/jobid/state` in `azure_client.out_cont_client`
    and stores them in the `shiny_cache` directory.
    If the cache path already exists for that run, nothing is downloaded.

    Raises ValueError error if `exp/jobid/state/` does not exist in the output blob
    """

    # will override files in cache
    shiny_cache_path = "shiny_visualizers/shiny_cache"
    if not os.path.exists(shiny_cache_path):
        os.makedirs(shiny_cache_path)
    azure_blob_path = os.path.join(exp, jobid, state)
    if scenario != "N/A":  # if visualizing a scenario append to the path
        azure_blob_path = os.path.join(azure_blob_path, scenario)
    azure_blob_path = azure_blob_path.replace("\\", "/") + "/"
    dest_path = os.path.join(shiny_cache_path, azure_blob_path)
    # if we already loaded this before, dont redownload it all!
    if os.path.exists(dest_path):
        return dest_path
    download_directory(
        azure_client.out_cont_client, azure_blob_path, shiny_cache_path
    )
    return dest_path


def _create_figure_from_timeline(
    timeline: pd.DataFrame,
    plot_name: str,
    x_axis: str,
    y_axis: Union[str, list[str]],
    plot_type: str,
    group_by: str = None,
):
    line = px.line(
        timeline, x=x_axis, y=y_axis, title=plot_name, labels=y_axis
    )
    for data in line.data:
        # if we pass a list of y_axis, we provide labels only for what
        # differs from label to label, this aids in readability
        if data.name == plot_type:
            new_name = data.name.replace("_", " ")
        else:
            new_name = data.name.replace(plot_type, "").replace("_", "-")
        data.name = new_name
        data.hovertemplate = new_name
    return line


def load_default_timelines(cache_path):
    timelines = pd.read_csv(
        os.path.join(cache_path, "azure_visualizer_timeline.csv")
    )
    # lowercase all columns to avoid caps issues
    timelines.columns = [col.lower() for col in timelines.columns]
    assert (
        "date" in timelines.columns
    ), "something went wrong in the creation of azure_visualizer_timeline.csv, there is no date columns"
    num_individual_particles = (
        len(timelines["chain_particle"].unique())
        if "chain_particle" in timelines.columns
        else 1
    )
    if num_individual_particles == 1:
        timelines["chain_particle"] = "na_na"
    # we are counting the number of plot_types that are within timelines.columns
    # this way we dont try to plot something that timelines does not have info on
    plots_in_timelines = [
        True if any([plot_type in col for col in timelines.columns]) else False
        for plot_type in OVERVIEW_PLOT_TYPES
    ]
    num_unique_plots_in_timelines = sum(plots_in_timelines)
    # select only the plots we actually find within `timelines`
    plot_types = OVERVIEW_PLOT_TYPES[plots_in_timelines].tolist()
    plot_titles = OVERVIEW_PLOT_TITLES[plots_in_timelines].tolist()

    # total_subplots = num_individual_particles * num_unique_plots
    fig = make_subplots(
        rows=num_unique_plots_in_timelines,
        cols=num_individual_particles,
        shared_xaxes=True,
        x_title="date",
        row_heights=[OVERVIEW_SUBPLOT_HEIGHT] * num_unique_plots_in_timelines,
        column_widths=[OVERVIEW_SUBPLOT_WIDTH] * num_individual_particles,
        subplot_titles=plot_titles,
    )

    for plot_num, (plot_title, plot_type) in enumerate(
        zip(plot_titles, plot_types), start=1
    ):
        # for example "vaccination_" in "vaccination_0_17" is true
        # so we include this column in the plot under that plot_type
        columns_to_plot = [
            col for col in timelines.columns if plot_type in col
        ]
        plot_by_chain_particle = _create_figure_from_timeline(
            timelines,
            plot_title,
            x_axis="date",
            y_axis=columns_to_plot,
            plot_type=plot_type,
            group_by="chain_particle",
        )
        # _create_figure_from_timeline gives us a figure obj, not quite what we need
        # TODO test this with multiple chains and see what happens to the shape of plot_by_chain_particle["data"]
        for chain_particle_idx in range(num_individual_particles):
            for group_by_num in range(len(columns_to_plot)):
                fig.add_trace(
                    plot_by_chain_particle["data"][group_by_num],
                    row=plot_num,
                    col=chain_particle_idx + 1,  # 1 start indexing, not 0 here
                )
    for i, yaxis in enumerate(fig.select_yaxes(), 1):
        legend_name = f"legend{i}"
        fig.update_layout(
            {legend_name: dict(y=yaxis.domain[1], yanchor="top")},
            showlegend=True,
        )
        fig.update_traces(row=i, legend=legend_name)
    fig.update_layout(
        width=OVERVIEW_SUBPLOT_WIDTH + 50,
        height=OVERVIEW_SUBPLOT_HEIGHT * num_unique_plots_in_timelines + 50,
        title_text="Overview",
        legend_tracegroupgap=20,
    )

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
default_job_list = [
    node.name
    for node in tree_root.children[experiment_names[0]].children.values()
]
default_state_list = [
    node.name
    for node in tree_root.children[experiment_names[0]]
    .children[default_job_list[0]]
    .children.values()
]
default_scenario_list = ["N/A"]
default_chain_particle_list = ["N/A"]
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
            ui.input_selectize(
                "chain_particle",
                "Chain+Particle",
                default_chain_particle_list,
                selected=default_chain_particle_list[0],
            ),
            ui.output_text("counter"),
        ),
        # ui.output_plot("plot", height="750px", click=True,),
        output_widget("plot"),
    ),
)


def server(input, output, session: Session):
    @reactive.effect
    @reactive.event(input.experiment, input.job_id, input.state)
    def _():
        """Updates the Buttons by traversing the tree of blob directories
        So the user is always seeing the correct directories"""
        experiment = input.experiment()
        tree_exp = tree_root.children[experiment]
        new_job_id_selections = [
            node.name for node in tree_exp.children.values()
        ]
        # update the jobs able to be picked based on the currently selected experiment
        selected_jobid = (
            input.job_id()
            if input.job_id() in new_job_id_selections
            else new_job_id_selections[0]
        )
        ui.update_selectize(
            "job_id", choices=new_job_id_selections, selected=selected_jobid
        )
        tree_job = tree_exp.children[selected_jobid]
        new_state_selections = [
            node.name for node in tree_job.children.values()
        ]
        selected_state = (
            input.state()
            if input.state() in new_state_selections
            else new_state_selections[0]
        )
        # update the states able to be picked based on the currently selected job
        ui.update_selectize(
            "state",
            choices=[node.name for node in tree_job.children.values()],
            selected=selected_state,
        )
        tree_state = tree_job.children[selected_state]
        # if our tree goes another lvl deep, we are dealing with scenarios here
        if list(tree_state.children.values())[0].children:
            new_scenario_selections = [
                node.name for node in tree_state.children.values()
            ]
            selected_scenario = (
                input.scenario()
                if input.scenario() in new_scenario_selections
                else new_scenario_selections[0]
            )
            ui.update_selectize(
                "scenario",
                choices=new_scenario_selections,
                selected=selected_scenario,
            )

    @output(id="plot")
    @render_widget
    @reactive.event(input.action_button)
    def counter():
        """
        Gets the files associated with that experiment+job+state+scenario combo
        and visualizes some summary statistics about the run
        """
        exp = input.experiment()
        job_id = input.job_id()
        state = input.state()
        scenario = input.scenario()
        cache_path = get_azure_files(
            exp, job_id, state, scenario, azure_client
        )

        fig = load_default_timelines(cache_path)
        return fig


app = App(app_ui, server)
app.run()
