"""
This Python script, when run, launches a Shiny app on http://localhost:8000/ that aids in visualizing
individual particles of posteriors produced by an experiment.
"""

# ruff: noqa: E402
import os
from typing import Iterable, Union

import numpy as np
import pandas as pd
import plotly
import plotly.express as px
from azure.core.paging import ItemPaged
from cfa_azure.clients import AzureClient
from cfa_azure.helpers import download_directory
from plotly.subplots import make_subplots
from shiny import App, Session, reactive, ui
from shinywidgets import output_widget, render_widget

from mechanistic_model.covid_initializer import CovidInitializer
from mechanistic_model.mechanistic_inferer import MechanisticInferer

INITIALIZER_USED = CovidInitializer
INFERER_USED = MechanisticInferer
INPUT_BLOB_NAME = "scenarios-mechanistic-input"
OUTPUT_BLOB_NAME = "scenarios-mechanistic-output"
# this will reduce the time it takes to load the azure connection, but only shows
# one experiment worth of data, which may be what you want...
#  leave empty ("") to explore all experiments
PRE_FILTER_EXPERIMENTS = "fifty_state_2304_2404_3strain"
# when loading the overview timelines csv for each run, columns
# are expected to have names corresponding to the type of plot they create
# vaccination_0_17 specifies the vaccination_ plot type, multiple columns may share
# a plot type, e.g: vaccination_0_17, vaccination_18_49,...
# it is assumed if they share a plot type they will appear on the same plot together
OVERVIEW_PLOT_TYPES = np.array(
    [
        "seasonality_coef",
        "vaccination_",
        "_external_introductions",
        "_strain_proportion",
        "_average_immunity",
        "total_infection_incidence",  # TODO MAKE AGE SPECIFIC
        "pred_hosp_",
    ]
)
# plot titles for each type of plot
OVERVIEW_PLOT_TITLES = np.array(
    [
        "Seasonality Coefficient",
        "Vaccination Rate By Age",
        "External Introductions by Strain",
        "Strain Proportion of New Infections",
        "Average Population Immunity Against Strains",
        "Total Infection Incidence",
        "Predicted Hospitalizations",
    ]
)
OVERVIEW_SUBPLOT_HEIGHT = 150
OVERVIEW_SUBPLOT_WIDTH = 1250


def build_azure_connection(
    config_path: str = "secrets/configuration_cfaazurebatchprd_new_sp.toml",
) -> AzureClient:
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


def get_blob_names(azure_client: AzureClient) -> ItemPaged[str]:
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
        self.subdirs = {}


def construct_tree(file_paths: Iterable[str]) -> Node:
    """given a iterable of strings, constructs a tree of directories from root "/".
    Used to efficiently traverse directories after tree is constructed.
    leaf nodes are files with a "." like .txt or .json

    Parameters
    ----------
    file_paths : Iterable[str]
        a list or iterable containing file paths, each file path may not begin with "/"
        any directories or paths of directories with no files at the end will be skipped.

    Returns
    -------
    Node
        A node object containing a dictionary of subdirectories, files have a `name` field but
        an empty dictionary of subdirs.
    """
    root = Node("/")
    for path in file_paths:
        # indicates this is an actual file like .txt or .json or .out
        if "." in path:
            directories, filename = path.rsplit("/", 1)
            current_node = root
            for directory in directories.split("/"):
                if directory not in current_node.subdirs:
                    current_node.subdirs[directory] = Node(directory)
                current_node = current_node.subdirs[directory]
            current_node.subdirs[filename] = Node(filename)
    return root


def get_azure_files(
    exp: str, jobid: str, state: str, scenario: str, azure_client: AzureClient
) -> str:
    """Reads in all files from the output blob `exp/jobid/state` in `azure_client.out_cont_client`
    and stores them in the `shiny_cache` directory.
    If the cache path already exists for that run (even if it is empty!), nothing is downloaded.

    Raises ValueError error if `exp/jobid/state/` does not exist in the output blob

    Parameters
    ----------
    exp : str
        experiment name
    jobid : str
        jobid name
    state : str
        usps postal code of the state
    scenario : str
        optional scenario, N/A if not applicable to the directory structure
    azure_client : cfa_azure.AzureClient
        Azure client to access azure blob storage and download the files

    Returns
    -------
    str
        path into which files were loaded (if they did not already exist there)
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
) -> plotly.graph_objs.Figure:
    """
    PRIVATE FUNCTION
    plots a line plot given a pandas dataframe and some x/y axis to plot and group bys.
    Formats this figure according to the plot_type title passed.

    Parameters
    ----------
    timeline : pd.DataFrame
        dataframe containing data to plot
    plot_name : str
        plot title
    x_axis : str
        title of the column which will serve as the x axis, usually "date"
    y_axis : Union[str, list[str]]
        title or titles of the columns to be plotted as y axis
    plot_type : str
        the generic name of the plot, if multiple y_axis names are provided,
        the diff of each y_axis name from the `plot_type` is used to identify each line
    group_by : str, optional
        a column name on which to group by, by default None

    Returns
    -------
    plotly.graph_objs.Figure
        figure used to append traces
    """
    line = px.line(
        timeline,
        x=x_axis,
        y=y_axis,
        title=plot_name,
        labels=y_axis,
        line_group=group_by,
    )
    for data in line.data:
        # if we pass a list of y_axis, we provide labels only for what
        # differs from label to label, this aids in readability
        if data.name == plot_type:
            new_name = data.name.replace("_", " ")
        else:
            new_name = data.name.replace(plot_type, "").replace("_", "-")
        data.name = new_name
        data.hovertemplate = "%{y:.3g}"
    return line


def load_checkpoint_inference_chains(cache_path) -> plotly.graph_objs.Figure:
    """Given a path a folder containing downloaded azure files, checks for the existence
    of the checkpoint.json file, if it exists, returns a figure plotting
    the inference chains of each of the sampled parameters, raises a FileNotFoundError if csv
    is not found within `cache_path`

    Parameters
    ----------
    cache_path : str
        path to the local path on machine with files within.

    Returns
    -------
    Figure
        plotly Figure with `n` rows and `m` columns where `n` is the number of columns
        within azure_visualizer_timeline identified by OVERVIEW_PLOT_TYPES global var.
    """
    return cache_path


def load_default_timelines(cache_path) -> plotly.graph_objs.Figure:
    """
    Given a path to a folder containing downloaded azure files, checks for the existence
    of the azure_visualizer_timeline csv, if it exists, returns an overview figure
    of the timelines provided in the csv, raises a FileNotFoundError if csv
    is not found within `cache_path`

    Parameters
    ----------
    cache_path : str
        path to the local path on machine with files within.

    Returns
    -------
    Figure
        plotly Figure with `n` rows and `m` columns where `n` is the number of columns
        within azure_visualizer_timeline identified by OVERVIEW_PLOT_TYPES global var.
    """
    timeline_path = os.path.join(cache_path, "azure_visualizer_timeline.csv")
    if not os.path.exists(timeline_path):
        raise FileNotFoundError(
            "attempted to visualize an overview without an `azure_visualizer_timeline.csv` file"
        )
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
    # rather than using row_titles which appear weirdly,
    # we will title only the left most plot of each row
    subplot_titles_spaced = [
        (
            plot_titles[int(i / num_individual_particles)]
            if i % num_individual_particles == 0
            else ""
        )
        for i in range(
            num_unique_plots_in_timelines * num_individual_particles
        )
    ]

    # total_subplots = num_individual_particles * num_unique_plots
    fig = make_subplots(
        rows=num_unique_plots_in_timelines,
        cols=num_individual_particles,
        shared_xaxes=True,
        shared_yaxes=True,
        x_title="date",
        row_heights=[OVERVIEW_SUBPLOT_HEIGHT] * num_unique_plots_in_timelines,
        column_widths=[OVERVIEW_SUBPLOT_WIDTH] * num_individual_particles,
        subplot_titles=subplot_titles_spaced,
        horizontal_spacing=0.01,
        vertical_spacing=0.03,
    )

    for plot_num, (plot_title, plot_type) in enumerate(
        zip(plot_titles, plot_types), start=1
    ):
        # for example "vaccination_" in "vaccination_0_17" is true
        # so we include this column in the plot under that plot_type
        columns_to_plot = [
            col for col in timelines.columns if plot_type in col
        ]
        # generates a big figure of the column(s) of interest, possibly grouped by chain_particle
        plot_by_chain_particle = _create_figure_from_timeline(
            timelines,
            plot_title,
            x_axis="date",
            y_axis=columns_to_plot,
            plot_type=plot_type,
            group_by="chain_particle",
        )
        # _create_figure_from_timeline gives us a figure obj, not quite what we need
        # we can query the "data" key to get access to each trace, but it flattens the columns and chain_particles
        # into a single list, so we need creative indexing to traverse the flattened list,
        # e.g. group_by_num * num_individual_particles + chain_particle_idx
        for chain_particle_idx in range(num_individual_particles):
            for col_num in range(len(columns_to_plot)):
                fig.add_trace(
                    plot_by_chain_particle["data"][
                        col_num * num_individual_particles + chain_particle_idx
                    ],
                    row=plot_num,
                    col=chain_particle_idx + 1,  # 1 start indexing, not 0 here
                )
    # here we are setting up one legend per row, it is messy but plotly does not
    # usually allow for subplot legends so here we are
    for i, yaxis in enumerate(fig.select_yaxes()):
        legend_name = f"legend{int(i / num_individual_particles) + 1}"
        if i % num_individual_particles == 0:
            fig.update_layout(
                {legend_name: dict(y=yaxis.domain[1], yanchor="top")},
                showlegend=True,
            )
            fig.update_traces(
                row=int(i / num_individual_particles) + 1,
                legend=legend_name,
            )
    fig.update_layout(
        width=OVERVIEW_SUBPLOT_WIDTH + 50,
        height=OVERVIEW_SUBPLOT_HEIGHT * num_unique_plots_in_timelines + 50,
        title_text="",
        legend_tracegroupgap=0,
        hovermode="x unified",
    )
    fig.update_annotations(font_size=16, x=0.0, xanchor="left")

    return fig


print("Connecting to Azure Storage")
azure_client = build_azure_connection()
print("Retrieving and Organizing Azure File Structure (approx 10 seconds)")
blobs = get_blob_names(azure_client)
tree_root = construct_tree(blobs)

# now that we have all the paths stored in a Tree, all these operations are quick
# these are used to initally populate the selectors
# and are then dynamically updated based on the experiment chosen
experiment_names = [node.name for node in tree_root.subdirs.values()]
default_job_list = [
    node.name
    for node in tree_root.subdirs[experiment_names[0]].subdirs.values()
]
default_state_list = [
    node.name
    for node in tree_root.subdirs[experiment_names[0]]
    .subdirs[default_job_list[0]]
    .subdirs.values()
]
default_scenario_list = ["N/A"]
default_chain_particle_list = ["N/A"]
####################################### SHINY CODE ################################################
app_ui = ui.page_fluid(
    ui.h2("Azure Visualizer"),
    ui.layout_sidebar(
        ui.sidebar(
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
            ui.input_switch("dark_mode", "Dark Mode", True),
            width=350,
        ),
        ui.navset_card_tab(
            ui.nav_panel("Overview", output_widget("plot")),
            ui.nav_panel(
                "Inference Chains",
                "TODO",  # output_widget("plot_inference_chains")
            ),
        ),
    ),
)


def server(input, output, session: Session):
    @reactive.effect
    @reactive.event(input.dark_mode)
    def _():
        """
        a simple function which toggles the background UI colors from light to dark mode
        """
        if input.dark_mode():
            ui.update_dark_mode("dark")
        else:
            ui.update_dark_mode("light")

    @reactive.effect
    @reactive.event(input.experiment, input.job_id, input.state)
    def _():
        """Updates the Buttons by traversing the tree of blob directories
        So the user is always seeing the correct directories"""
        experiment = input.experiment()
        tree_exp = tree_root.subdirs[experiment]
        new_job_id_selections = [
            node.name for node in tree_exp.subdirs.values()
        ]
        if len(new_job_id_selections) == 0:
            raise RuntimeError("This experiment contains no job folders!")
        # update the jobs able to be picked based on the currently selected experiment
        selected_jobid = (
            input.job_id()
            if input.job_id() in new_job_id_selections
            else new_job_id_selections[0]
        )
        ui.update_selectize(
            "job_id", choices=new_job_id_selections, selected=selected_jobid
        )
        # select the job directory and lookup the possible states run in that job
        tree_job = tree_exp.subdirs[selected_jobid]
        new_state_selections = [
            node.name for node in tree_job.subdirs.values()
        ]
        if len(new_state_selections) == 0:
            raise RuntimeError("This job contains no state folders!")
        selected_state = (
            input.state()
            if input.state() in new_state_selections
            else new_state_selections[0]
        )
        # update the states able to be picked based on the currently selected job
        ui.update_selectize(
            "state",
            choices=[node.name for node in tree_job.subdirs.values()],
            selected=selected_state,
        )
        tree_state = tree_job.subdirs[selected_state]
        # if our tree goes another level deep, we are dealing with scenarios here
        # if the subdir of the current state itself has subdirs, we assume those are scenarios
        if len(tree_state.subdirs.values()) == 0:
            raise RuntimeError(
                "No files/directories found within the state folder!"
            )
        if list(tree_state.subdirs.values())[0].subdirs:
            new_scenario_selections = [
                node.name for node in tree_state.subdirs.values()
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
    def _():
        """
        Gets the files associated with that experiment+job+state+scenario combo
        and visualizes some summary statistics about the run
        """
        # read in the directory path
        exp = input.experiment()
        job_id = input.job_id()
        state = input.state()
        scenario = input.scenario()
        # get files and store them localy
        cache_path = get_azure_files(
            exp, job_id, state, scenario, azure_client
        )
        # read in the timelines.csv if it exists, and load the figure, error if it doesnt exist
        fig = load_default_timelines(cache_path)
        # we have the figure, now update the light/dark mode depending on the switch
        dark_mode = input.dark_mode()
        if dark_mode:
            theme = "plotly_dark"
        else:
            theme = "plotly_white"
        fig.update_layout(template=theme)
        return fig

    @output(id="plot_inference_chains")
    @render_widget
    @reactive.event(input.action_button)
    def _():
        """
        Gets the files associated with that experiment+job+state+scenario combo
        and visualizes the inference chains on the inference run (if applicable)
        """
        exp = input.experiment()
        job_id = input.job_id()
        state = input.state()
        scenario = input.scenario()
        cache_path = get_azure_files(
            exp, job_id, state, scenario, azure_client
        )

        fig = load_checkpoint_inference_chains(cache_path)
        return fig


app = App(app_ui, server)
app.run()
