import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from shiny import App, render, ui

import utils
from config.config_base import ConfigBase
from mechanistic_compartments import build_basic_mechanistic_model

model = build_basic_mechanistic_model(ConfigBase())
(s_compartment, e_compartment, i_compartment, _) = model.INITIAL_STATE
age_choices = ["All"] + model.AGE_GROUP_STRS
# indexes for the age bins
age_dict = {
    age_string: idx for idx, age_string in enumerate(model.AGE_GROUP_STRS)
}
# index for all category
age_dict["All"] = list(range(model.NUM_AGE_GROUPS))
vaccination_strings = ["0", "1", "2+"]
immune_state_strings = [
    "No Prior Exposure",
    "pre-omicron only",
    "omicron only",
    "both",
]
compartment_choices = [
    "Susceptible",
    "Exposed",
    "Infectious",
]
compartment_dict = {
    "Susceptible": s_compartment,
    "Exposed": e_compartment,
    "Infectious": i_compartment,
}


app_ui = ui.page_fixed(
    ui.h2("Visualizing Immune History"),
    ui.markdown(
        """
    """
    ),
    ui.layout_sidebar(
        ui.panel_sidebar(
            ui.input_selectize(
                "compartment",
                "Compartment",
                compartment_choices,
                multiple=True,
                selected="Susceptible",
            ),
            ui.input_selectize(
                "age_bin",
                "Age Bins",
                age_choices,
                multiple=True,
                selected="All",
            ),
            ui.input_selectize(
                "display",
                "Values",
                ["Proportion", "Count"],
                selected="Proportion",
            ),
        ),
        ui.panel_main(ui.output_plot("plot", height="750px")),
    ),
)


def heatmap(input, fig, ax):
    """
    a helper function which takes input and generates a heatmap inside of ax
    returning the figure and axis objects modified with the heatmap included.

    """
    if (
        len(input.compartment()) == 0
        or len(input.age_bin()) == 0
        or len(input.display()) == 0
    ):
        return None
    compartment_selections = input.compartment()
    compartment = compartment_dict[compartment_selections[0]]
    compartment = np.sum(compartment, axis=-1)  # sum across wane/strain
    if len(compartment_selections) > 1:
        for additional_compartment_name in compartment_selections[1:]:
            additional_compartment = compartment_dict[
                additional_compartment_name
            ]
            # make shapes line up, sum across wane/strain dimension
            compartment += np.sum(additional_compartment, axis=-1)
    age_selections = input.age_bin()
    age_bin = []
    for age in age_selections:
        if age == "All":
            age_bin = age_dict[age]
            break
        else:
            age_bin.append(age_dict[age])
    # age_bin = age_dict[input.age_bin()[0]]
    compartment = compartment[age_bin, :, :]
    if isinstance(age_bin, list):
        compartment = np.sum(compartment, axis=0)
    format_string = ".0f"
    if input.display() == "Proportion":
        compartment = compartment / np.sum(compartment)
        format_string = "0.1%"
    heatmap = sn.heatmap(
        compartment, linewidth=0.5, ax=ax, annot=True, fmt=format_string
    )
    heatmap.set_xticklabels(vaccination_strings)
    heatmap.tick_params(axis="y", labelrotation=45)
    heatmap.set_yticklabels(immune_state_strings)
    heatmap.set_xlabel("Vaccination Count")
    heatmap.set_ylabel("Immune History")
    heatmap.set_title(
        "Population of "
        + "+".join(compartment_selections)
        + " compartment(s) stratified by immune hist and vaccination"
    )
    return fig, ax


def waning_in_population(input, fig, ax):
    compartment_names = input.compartment()
    if (
        len(compartment_names) == 0
        or len(compartment_names) > 1
        or compartment_names[0] != "Susceptible"
    ):
        return fig, ax  # this plot only works on the Suceptible compartment
    age_selections = input.age_bin()
    age_bin = []
    for age in age_selections:
        if age == "All":
            age_bin = age_dict[age]
            break
        else:
            age_bin.append(age_dict[age])
    age_strings = [model.AGE_GROUP_STRS[abin] for abin in age_bin]
    compartment = s_compartment[age_bin, :, :, :]
    if input.display() == "Proportion":
        compartment = compartment / np.sum(compartment)
    # if we only looking at one age bin we still want a separate dimension so code dont break
    if len(age_bin) == 1:
        compartment.reshape([1] + list(compartment.shape))
    immune_compartments = [
        compartment[:, :, :, w_idx] for w_idx in model.W_IDX
    ]
    # reverse for plot readability since we read left to right
    immune_compartments = np.array(immune_compartments[::-1])
    x_axis = ["W" + str(int(idx)) for idx in model.W_IDX][::-1]
    age_to_immunity_slice = {}
    # for each age group, plot its number of persons in each immune compartment
    # stack the bars on top of one another by summing the previous age groups underneath
    for age_idx, age_group in zip(age_bin, age_strings):
        age_to_immunity_slice[age_group] = np.sum(
            immune_compartments[:, age_idx, :, :], axis=(-1, -2)
        )
        ax.bar(
            x_axis,
            age_to_immunity_slice[age_group],
            label=age_group,
            bottom=sum(
                [age_to_immunity_slice[x] for x in age_strings[0:age_idx]]
            ),
        )
    # props = {"rotation": 25, "size": 7}
    # plt.setp(ax.get_xticklabels(), **props)
    ax.legend()
    ax.tick_params(axis="x", labelrotation=25)
    ax.set_title("Initial Population Immunity level by waning compartment")
    ax.set_xlabel("Immune Compartment")
    ax.set_ylabel("Population " + input.display())
    return fig, ax


def model_sero_curve(input, fig, ax):
    """
    A function that plots the hypothetical serology curve of the population up to the initialization date.

    Parameters
    ----------
    `input`:
        an object holding the Shiny app input parameters, such as the compartments selected and age groups specified.
    `fig`: plt.fig
        matplotlib.pyplot figure object
    `ax`: plt.ax
        matplotlib.axis object where the plot will be drawn on

    Returns
    ----------
    the `fig` and `ax` parameters, modified to have the model sero curve drawn in.
    """

    abm_population = pd.read_csv(model.SIM_DATA)
    # remove those with active infections, those are designated for exposed/infected
    if (
        len(input.compartment()) == 0
        or len(input.age_bin()) == 0
        or len(input.display()) == 0
    ):
        return None
    compartment_selections = input.compartment()
    # these selectors will help us narrow down our abm_population into just the groups we need.
    select_susceptible = abm_population["TSLIE"] >= 0
    select_exposed = (abm_population["TSLIE"] < 0) & (
        abm_population["infectious"] == 0
    )
    select_infected = (abm_population["TSLIE"] < 0) & (
        abm_population["infectious"] == 1
    )
    select_dict = {
        "Susceptible": select_susceptible,
        "Exposed": select_exposed,
        "Infectious": select_infected,
    }
    # build up the compartment filter using OR statements.
    compartment_filter = False
    for compartment in compartment_selections:
        compartment_filter = compartment_filter | select_dict[compartment]

    # filter to just our compartments
    abm_population = abm_population[compartment_filter]
    # next we want to add bins to our abm data just like our model does.
    abm_population = utils.prep_abm_data(
        abm_population,
        model.MAX_VAX_COUNT,
        model.AGE_LIMITS,
        model.WANING_TIMES,
        model.NUM_STRAINS,
        model.STRAIN_IDX,
    )
    # next filter down the age bins we want using the new columns we added above
    age_selections = input.age_bin()
    age_bin = []
    for age in age_selections:
        if age == "All":
            age_bin = age_dict[age]
            break
        else:
            age_bin.append(age_dict[age])
    abm_population = abm_population[abm_population["age_bin"].isin(age_bin)]


def server(input, output, session):
    @output
    @render.plot
    def plot():
        fig, axs = plt.subplots(2, 1)
        fig, axs[0] = heatmap(input, fig, axs[0])
        fig, axs[1] = waning_in_population(input, fig, axs[1])
        return fig


app = App(app_ui, server)
app.run()