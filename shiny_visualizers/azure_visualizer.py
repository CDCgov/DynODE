"""
This Python script, when run, launches a Shiny app on http://localhost:8000/ that aids in visualizing
individual particles of posteriors produced by an experiment.
"""

# ruff: noqa: E402
import numpy as np
from shiny import App, Session, reactive, ui
from shinywidgets import output_widget, render_widget

import shiny_visualizers.shiny_utils as sutils
from mechanistic_model.covid_initializer import CovidInitializer
from mechanistic_model.mechanistic_inferer import MechanisticInferer

INITIALIZER_USED = CovidInitializer
INFERER_USED = MechanisticInferer
INPUT_BLOB_NAME = "scenarios-mechanistic-input"
OUTPUT_BLOB_NAME = "scenarios-mechanistic-output"
# the path on your local machine where local projects are read in and azure data is stored
SHINY_CACHE_PATH = "shiny_visualizers/shiny_cache"
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


print("Connecting to Azure Storage")
azure_client = sutils.build_azure_connection(
    input_blob_name=INPUT_BLOB_NAME, output_blob_name=OUTPUT_BLOB_NAME
)
print("Retrieving and Organizing Azure File Structure (approx 10 seconds)")
blobs = sutils.get_blob_names(
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
                "chain_particle",
                "Chain+Particle",
                default_chain_particle_list,
                selected=default_chain_particle_list[0],
            ),
            ui.input_dark_mode(id="dark_mode", mode="dark"),
            # ui.input_switch("dark_mode", "Dark Mode", True),
            width=450,
        ),
        ui.navset_card_tab(
            ui.nav_panel("Overview", output_widget("plot")),
            ui.nav_panel(
                "Inference Chains", output_widget("plot_inference_chains")
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
    @reactive.event(input.experiment, input.job_id, input.states)
    def _():
        """Updates the Buttons by traversing the tree of blob directories
        So the user is always seeing the correct directories"""
        # get the current choices of the user, these may not actually be valid though
        # since the user may be actively updating a top level dir without changing bottom ones yet
        experiment = input.experiment()
        job_id = input.job_id()
        states = input.states()
        scenario = input.scenario()
        tree_exp = output_blob.subdirs[experiment]
        job_id_selections = [job.name for job in tree_exp.subdirs.values()]
        if len(job_id_selections) == 0:
            raise RuntimeError("This experiment contains no job folders!")
        # update the jobs able to be picked based on the currently selected experiment
        selected_jobid = (
            job_id if job_id in job_id_selections else job_id_selections[0]
        )
        ui.update_selectize(
            "job_id", choices=job_id_selections, selected=selected_jobid
        )
        # select the job directory and lookup the possible states run in that job
        tree_job = tree_exp.subdirs[selected_jobid]
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
        for state in selected_states:
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

    @output(id="plot")
    @render_widget
    @reactive.event(input.action_button)
    def _():
        """
        Gets the files associated with each selected experiment+job+state+scenario combo
        and visualizes some summary statistics about the run
        """
        # read in the directory path
        exp = input.experiment()
        job_id = input.job_id()
        states = input.states()
        scenario = input.scenario()
        # get files and store them localy
        cache_paths = sutils.get_azure_files(
            exp, job_id, states, scenario, azure_client, SHINY_CACHE_PATH
        )
        try:
            # read in the timelines.csv if it exists, and load the figure, error if it doesnt exist
            fig = sutils.load_default_timelines(
                cache_paths,
                states,
                plot_titles=OVERVIEW_PLOT_TITLES,
                plot_types=OVERVIEW_PLOT_TYPES,
                overview_subplot_height=OVERVIEW_SUBPLOT_HEIGHT,
                overview_subplot_width=OVERVIEW_SUBPLOT_WIDTH,
            )
        except FileNotFoundError as e:
            raise e
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
        try:
            fig = sutils.load_checkpoint_inference_chains(
                cache_paths,
                overview_subplot_height=OVERVIEW_SUBPLOT_HEIGHT,
                overview_subplot_width=OVERVIEW_SUBPLOT_WIDTH,
            )
        except FileNotFoundError as e:
            raise e
        # we have the figure, now update the light/dark mode depending on the switch
        dark_mode = input.dark_mode()
        if dark_mode:
            theme = "plotly_dark"
        else:
            theme = "plotly_white"
        fig.update_layout(template=theme)
        return fig

    @output(id="plot_sample_correlations")
    @render_widget
    @reactive.event(input.action_button)
    def _():
        """
        Gets the files associated with each selected experiment+job+states+scenario combo
        and visualizes the inference correlations on the inference run (if applicable)
        """
        exp = input.experiment()
        job_id = input.job_id()
        states = input.states()
        scenario = input.scenario()
        cache_paths = sutils.get_azure_files(
            exp, job_id, states, scenario, azure_client
        )
        try:
            fig = sutils.load_checkpoint_inference_correlations(
                cache_paths,
                overview_subplot_size=OVERVIEW_SUBPLOT_WIDTH,
            )
        except FileNotFoundError as e:
            raise e
        # we have the figure, now update the light/dark mode depending on the switch
        dark_mode = input.dark_mode()
        if dark_mode:
            theme = "plotly_dark"
        else:
            theme = "plotly_white"
        fig.update_layout(template=theme)
        return fig

    @output(id="plot_sample_violins")
    @render_widget
    @reactive.event(input.action_button)
    def _():
        """
        Gets the files associated with that experiment+job+state+scenario combo
        and visualizes the sampled values of the inference run as violin plots (if applicable)
        """
        exp = input.experiment()
        job_id = input.job_id()
        states = input.states()
        scenario = input.scenario()
        cache_paths = sutils.get_azure_files(
            exp, job_id, states, scenario, azure_client
        )
        try:
            fig = sutils.load_checkpoint_inference_violin_plots(
                cache_paths,
                overview_subplot_size=OVERVIEW_SUBPLOT_WIDTH,
            )
        except FileNotFoundError as e:
            raise e
        # we have the figure, now update the light/dark mode depending on the switch
        dark_mode = input.dark_mode()
        if dark_mode:
            theme = "plotly_dark"
        else:
            theme = "plotly_white"
        fig.update_layout(template=theme)
        return fig


app = App(app_ui, server)
app.run()
