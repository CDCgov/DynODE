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


def build_azure_connection(
    config_path: str = "secrets/configuration_cfaazurebatchprd_new_sp.toml",
    input_blob_name: str = "scenarios-mechanistic-input",
    output_blob_name: str = "scenarios-mechanistic-output",
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
        # hover template rounds to 4 sig figs
        data.hovertemplate = "%{y:.4g}"
    return line


def load_checkpoint_inference_chains(cache_path) -> plotly.graph_objs.Figure:
    """
    NOT YET IMPLEMENTED
    Given a path a folder containing downloaded azure files, checks for the existence
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


def _generate_row_wise_legends(fig, num_cols):
    # here we are setting up one legend per row, it is messy but plotly does not
    # usually allow for subplot row legends so we have to go through
    # each yaxis in the whole plot and do it this way
    for i, yaxis in enumerate(fig.select_yaxes()):
        # ruw and col values in plotly start at 1...
        row_num = int(i / num_cols) + 1
        legend_name = f"legend{row_num}"
        # we only want to do this operation once per row, thus % == 0
        if i % num_cols == 0:
            fig.update_layout(
                {legend_name: dict(y=yaxis.domain[1], yanchor="top")},
                showlegend=True,
            )
            # applys over all columns in this row
            fig.update_traces(
                row=row_num,
                legend=legend_name,
            )


def load_default_timelines(
    cache_path: str,
    plot_types: np.ndarray[str],
    plot_titles: np.ndarray[str],
    overview_subplot_width: int,
    overview_subplot_height: int,
) -> plotly.graph_objs.Figure:
    """
    Given a path to a folder containing downloaded azure files, checks for the existence
    of the azure_visualizer_timeline csv, if it exists, returns an overview figure
    of the timelines provided in the csv, raises a FileNotFoundError if csv
    is not found within `cache_path`

    Parameters
    ----------
    cache_path : str
        path to the local path on machine with files within.
    plot_types: np.ndarray[str]
        numpy array of strings representing all the plot types able to be plotted
        if they are not found within azure_visualizer_timeline they are skipped
    plot_titles: np.ndarray[str]
        numpy array of strings parallel to `plot_types` representing the plot titles
        in a more human readable format
    overview_subplot_width: int
        integer representing pixel width of each subplot in the overview
    overview_subplot_height: int
        integer representing pixel height of each subplot in the overview

    Returns
    -------
    Figure
        plotly Figure with `n` rows and `m` columns where `n` is the number of plots
        within azure_visualizer_timeline identified by OVERVIEW_PLOT_TYPES global var.
    """
    timeline_path = os.path.join(cache_path, "azure_visualizer_timeline.csv")
    if not os.path.exists(timeline_path):
        raise FileNotFoundError(
            "attempted to visualize an overview without an `azure_visualizer_timeline.csv` file"
        )
    timelines = pd.read_csv(timeline_path)
    assert (
        "date" in timelines.columns
    ), "something went wrong in the creation of azure_visualizer_timeline.csv, there is no date column"
    # count the `chain_particle` column, if it exists,
    # to figure out how many particles we are working with
    # if the column does not exist, na_na as placeholder
    if "chain_particle" not in timelines.columns():
        timelines["chain_particle"] = "na_na"

    num_individual_particles = len(timelines["chain_particle"].unique())
    # we are counting the number of plot_types that are within timelines.columns
    # this way we dont try to plot something that timelines does not have
    plots_in_timelines = [
        any([plot_type in col for col in timelines.columns])
        for plot_type in plot_types
    ]
    num_unique_plots_in_timelines = sum(plots_in_timelines)
    # select only the plots we actually find within `timelines`
    plot_types = plot_types[plots_in_timelines].tolist()
    plot_titles = plot_titles[plots_in_timelines].tolist()
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
    # generate subplots with some basic settings
    fig = make_subplots(
        rows=num_unique_plots_in_timelines,
        cols=num_individual_particles,
        shared_xaxes=True,
        shared_yaxes=True,
        x_title="date",
        row_heights=[overview_subplot_height] * num_unique_plots_in_timelines,
        column_widths=[overview_subplot_width] * num_individual_particles,
        subplot_titles=subplot_titles_spaced,
        horizontal_spacing=0.01,
        vertical_spacing=0.03,
    )

    # go through each plot type, look for matching columns within `timlines` and plot
    # that plot_type for each chain_particle pair. plotly rows/cols are index at 1 not 0
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
    _generate_row_wise_legends(fig, num_individual_particles)
    # lastly we update the whole figure's width and height with some padding
    fig.update_layout(
        width=overview_subplot_width + 50,
        height=overview_subplot_height * num_unique_plots_in_timelines + 50,
        title_text="",
        legend_tracegroupgap=0,
        hovermode="x unified",
    )
    # this is for the row titles font and position
    fig.update_annotations(font_size=16, x=0.0, xanchor="left")

    return fig
