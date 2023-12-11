import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
from shiny import App, render, ui

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
