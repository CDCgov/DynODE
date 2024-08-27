"""
This Python script, when run, launches a Shiny app on http://localhost:8000/ that aids in visualizing
individual particles of posteriors produced by an experiment.
"""

import os

# ruff: noqa: E402
import numpy as np
import pandas as pd
from plotly.graph_objects import Figure
from shiny import App, Session, reactive, ui
from shinywidgets import output_widget, render_plotly, render_widget

import mechanistic_azure.azure_utilities as autils
import shiny_visualizers.shiny_utils as sutils

INPUT_BLOB_NAME = "scenarios-mechanistic-input"
OUTPUT_BLOB_NAME = "scenarios-mechanistic-output"
# the path on your local machine where local projects are read in and azure data is stored
SHINY_CACHE_PATH = "shiny_visualizers/shiny_cache"
# this will reduce the time it takes to load the azure connection, but only shows
# one experiment worth of data, which may be what you want...
#  leave empty ("") to explore all experiments
PRE_FILTER_EXPERIMENTS = ""
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
        "External Introductions by Strain (per 100k)",
        "Strain Proportion of New Infections",
        "Average Population Immunity Against Strains",
        "Total Infection Incidence (per 100k)",
        "Predicted Hospitalizations (per 100k)",
    ]
)
# some plots need to be normalized per 100k, some dont
OVERVIEW_PLOT_Y_AXIS_NORMALIZATION = np.array(
    [1, 1, 100000, 1, 1, 100000, 100000]
)
# how often should metrics be reported, every x days?
# lower values heavily impact runtime, higher values may cause things to be skipped.
OVERVIEW_DAY_FIDELITY = 3
OVERVIEW_SUBPLOT_HEIGHT = 150
OVERVIEW_SUBPLOT_WIDTH = 550
DEMOGRAPHIC_DATA_PATH = "data/demographic-data/"
STATE_NAME_LOOKUP = pd.read_csv(
    os.path.join(DEMOGRAPHIC_DATA_PATH, "locations.csv")
)
POP_COUNTS_LOOKUP = pd.read_csv(
    os.path.join(DEMOGRAPHIC_DATA_PATH, "CenPop2020_Mean_ST.csv")
)

print("Connecting to Azure Storage")
azure_client = autils.build_azure_connection(
    input_blob_name=INPUT_BLOB_NAME, output_blob_name=OUTPUT_BLOB_NAME
)
print("Retrieving and Organizing Azure File Structure (approx 10 seconds)")
blobs = autils.get_blob_names(
    azure_client, name_starts_with=PRE_FILTER_EXPERIMENTS
)
output_blob = sutils.construct_tree(blobs)
print("accessing any local projects stored inside of %s" % SHINY_CACHE_PATH)
output_blob = sutils.append_local_projects_to_tree(
    SHINY_CACHE_PATH, output_blob
)

# now that we have all the paths stored in a Tree, all these operations are quick
# these are used to initally populate the selectors
# and are then dynamically updated based on the experiment chosen
experiment_names = [dir.name for dir in output_blob.subdirs.values()]
default_job_list = [
    dir.name
    for dir in output_blob.subdirs[experiment_names[0]].subdirs.values()
]
default_states_list = [
    dir.name
    for dir in output_blob.subdirs[experiment_names[0]]
    .subdirs[default_job_list[0]]
    .subdirs.values()
]
default_scenario_list = ["N/A"]
default_chain_particle_list = ["N/A"]
####################################### SHINY CODE ################################################
app_ui = ui.page_fluid(
    ui.h2("Azure Visualizer, work in progress"),
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
                "states",
                "States",
                default_states_list,
                multiple=True,
                options={"plugins": ["clear_button"]},
                selected=default_states_list[0],
            ),
            ui.input_selectize(
                "scenario",
                "Scenario",
                default_scenario_list,
                selected=default_scenario_list[0],
            ),
            ui.input_action_button("action_button", "Visualize Output"),
            ui.input_selectize(
                "overview_cols",
                "Overview Columns",
                OVERVIEW_PLOT_TITLES.tolist(),
                selected=OVERVIEW_PLOT_TITLES.tolist(),
                multiple=True,
            ),
            ui.input_dark_mode(id="dark_mode", mode="dark"),
            # ui.input_switch("dark_mode", "Dark Mode", True),
            width=450,
        ),
        ui.navset_card_tab(
            ui.nav_panel(
                "Overview",
                output_widget(
                    "plot_overview",
                    width=OVERVIEW_SUBPLOT_WIDTH * 50,
                ),
            ),
            ui.nav_panel(
                "Inference Chains",
                output_widget(
                    "plot_inference_chains",
                    width=OVERVIEW_SUBPLOT_WIDTH * 7,
                ),
            ),
            ui.nav_panel(
                "Sample Correlations",
                output_widget("plot_sample_correlations"),
            ),
            ui.nav_panel(
                "Sample Violin Plots",
                output_widget("plot_sample_violins"),
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
        ui.update_dark_mode(input.dark_mode())

    @reactive.effect
    @reactive.event(input.experiment)
    def update_jobs_from_experiment_selection():
        experiment = input.experiment()
        job_id = input.job_id()
        tree_exp = output_blob.subdirs[experiment]
        job_id_selections = [job.name for job in tree_exp.subdirs.values()]
        if len(job_id_selections) == 0:
            raise RuntimeError("This experiment contains no job folders!")
        selected_jobid = (
            job_id if job_id in job_id_selections else job_id_selections[0]
        )
        ui.update_selectize(
            "job_id", choices=job_id_selections, selected=selected_jobid
        )

    @reactive.effect
    @reactive.event(input.job_id)
    def update_states_from_job_id_selection():
        # we are given experiment/job_id to be valid selections
        experiment = input.experiment()
        job_id = input.job_id()
        states = input.states()
        tree_job = output_blob.subdirs[experiment].subdirs[job_id]
        possible_state_selections = [
            state.name for state in tree_job.subdirs.values()
        ]
        # pick any states user have selected, as long as they are valid selections
        selected_states = [
            state for state in states if state in possible_state_selections
        ]
        # if none of the users previous selections are valid, default back to the first
        if not selected_states:
            selected_states = [possible_state_selections[0]]
        # update the states able to be picked based on the currently selected job
        ui.update_selectize(
            "states",
            choices=possible_state_selections,
            selected=selected_states,
        )

    @reactive.effect
    @reactive.event(input.states)
    def update_scenarios_from_state_selection():
        # we are given experiment/job_id/states to be valid selections
        # because the above methods have already triggered
        experiment = input.experiment()
        job_id = input.job_id()
        states = input.states()
        scenario = input.scenario()
        tree_job = output_blob.subdirs[experiment].subdirs[job_id]
        for state in states:
            tree_state = tree_job.subdirs[state]

            # peak inside the state to ensure there is something in there...
            if len(tree_state.subdirs.values()) == 0:
                raise RuntimeError(
                    "No files/directories found within the state folder!"
                )
            # if our tree goes another level deep, we are dealing with scenarios here
            # if the subdir of the current state itself has subdirs, we assume those are scenarios
            if list(tree_state.subdirs.values())[0].subdirs:
                new_scenario_selections = [
                    node.name for node in tree_state.subdirs.values()
                ]
                selected_scenario = (
                    scenario
                    if scenario in new_scenario_selections
                    else new_scenario_selections[0]
                )
                ui.update_selectize(
                    "scenario",
                    choices=new_scenario_selections,
                    selected=selected_scenario,
                )
            else:  # no scenarios, make sure we set it back to N/A list
                ui.update_selectize(
                    "scenario",
                    choices=default_scenario_list,
                    selected=default_scenario_list[0],
                )

    @output(id="plot_overview")
    @render_plotly
    @reactive.event(input.action_button)
    def plot_overview() -> Figure:
        """
        Gets the files associated with each selected experiment+job+state+scenario combo
        and visualizes some summary statistics about the run
        """
        # read in the directory path
        exp = input.experiment()
        job_id = input.job_id()
        states = input.states()
        scenario = input.scenario()
        # user selected some columns they want out of all selections
        selected_columns = input.overview_cols()
        # get indexes of selected columns to pass correct parallel arrays as well
        selected_columns_indexes = [
            i
            for i, col in enumerate(OVERVIEW_PLOT_TITLES)
            if col in selected_columns
        ]
        # get requested state files and store them localy
        cache_paths = sutils.get_azure_files(
            exp, job_id, states, scenario, azure_client, SHINY_CACHE_PATH
        )
        # lookup state population sizes for normalization per 100k
        pop_sizes = sutils.get_population_sizes(
            states, STATE_NAME_LOOKUP, POP_COUNTS_LOOKUP
        )
        # read in the timelines.csv if it exists, and load the figure, error if it doesnt exist
        fig = sutils.load_default_timelines(
            cache_paths,
            states,
            state_pop_sizes=pop_sizes,
            day_fidelity=OVERVIEW_DAY_FIDELITY,
            plot_titles=OVERVIEW_PLOT_TITLES[selected_columns_indexes],
            plot_types=OVERVIEW_PLOT_TYPES[selected_columns_indexes],
            plot_normalizations=OVERVIEW_PLOT_Y_AXIS_NORMALIZATION[
                selected_columns_indexes
            ],
            overview_subplot_height=OVERVIEW_SUBPLOT_HEIGHT,
            overview_subplot_width=OVERVIEW_SUBPLOT_WIDTH,
        )
        # we have the figure, now update the light/dark mode depending on the switch
        theme = sutils.shiny_to_plotly_theme(input.dark_mode())
        fig.update_layout(template=theme)
        return fig

    @output(id="plot_inference_chains")
    @render_widget
    @reactive.event(input.action_button)
    def plot_inference_chains():
        """
        Gets the files associated with each selected experiment+job+state+scenario combo
        and visualizes the inference chains on the inference run (if applicable)
        """
        exp = input.experiment()
        job_id = input.job_id()
        states = input.states()
        scenario = input.scenario()
        cache_paths = sutils.get_azure_files(
            exp, job_id, states, scenario, azure_client, SHINY_CACHE_PATH
        )
        # only load first state for chains
        fig = sutils.load_checkpoint_inference_chains(
            cache_paths[0],
            overview_subplot_height=OVERVIEW_SUBPLOT_HEIGHT,
            overview_subplot_width=OVERVIEW_SUBPLOT_WIDTH,
        )
        # we have the figure, now update the light/dark mode depending on the switch
        theme = sutils.shiny_to_plotly_theme(input.dark_mode())
        fig.update_layout(template=theme)
        return fig

    @output(id="plot_sample_correlations")
    @render_widget
    @reactive.event(input.action_button)
    def plot_sample_correlations():
        """
        Gets the files associated with each selected experiment+job+states+scenario combo
        and visualizes the inference correlations on the inference run (if applicable)
        """
        exp = input.experiment()
        job_id = input.job_id()
        states = input.states()
        scenario = input.scenario()
        cache_paths = sutils.get_azure_files(
            exp, job_id, states, scenario, azure_client, SHINY_CACHE_PATH
        )
        # we have the figure, now update the light/dark mode depending on the switch
        fig = sutils.load_checkpoint_inference_correlations(
            cache_paths[0],
            overview_subplot_size=1500,
        )
        # we have the figure, now update the light/dark mode depending on the switch
        theme = sutils.shiny_to_plotly_theme(input.dark_mode())
        fig.update_layout(template=theme)
        return fig

    @output(id="plot_sample_violins")
    @render_widget
    @reactive.event(input.action_button)
    def plot_sample_violins():
        """
        Gets the files associated with that experiment+job+state+scenario combo
        and visualizes the sampled values of the inference run as violin plots (if applicable)
        """
        exp = input.experiment()
        job_id = input.job_id()
        states = input.states()
        scenario = input.scenario()
        cache_paths = sutils.get_azure_files(
            exp, job_id, states, scenario, azure_client, SHINY_CACHE_PATH
        )
        # we have the figure, now update the light/dark mode depending on the switch
        fig = sutils.load_checkpoint_inference_violin_plots(
            cache_paths[0],
            overview_subplot_size=1500,
        )
        # we have the figure, now update the light/dark mode depending on the switch
        theme = sutils.shiny_to_plotly_theme(input.dark_mode())
        fig.update_layout(template=theme)
        return fig


app = App(app_ui, server)
app.run()
