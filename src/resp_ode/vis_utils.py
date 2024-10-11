"""A series of utility functions for generating visualizations for the model"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _cleanup_and_normalize_timelines(
    all_state_timelines,
    plot_types,
    plot_normalizations,
    states,
    state_pop_sizes,
):
    # Select columns with 'float64' dtype
    float_cols = list(all_state_timelines.select_dtypes(include="float64"))
    # round down near-zero values to zero to make plots cleaner
    all_state_timelines[float_cols] = all_state_timelines[float_cols].mask(
        np.isclose(all_state_timelines[float_cols], 0, atol=1e-4), 0
    )
    for plot_type, plot_normalization in zip(plot_types, plot_normalizations):
        for state_name, state_pop in zip(states, state_pop_sizes):
            # if normalization is set to 1, we dont normalize at all.
            normalization_factor = (
                plot_normalization / state_pop
                if plot_normalization > 1
                else 1.0
            )
            # select all columns from that column type
            cols = [
                col for col in all_state_timelines.columns if plot_type in col
            ]
            # update that states columns by the normalization factor chosen for that column
            all_state_timelines.loc[
                all_state_timelines["state"] == state_name,
                cols,
            ] *= normalization_factor
    return all_state_timelines


def generate_model_overview_subplot_matplotlib(
    timeseries_df: pd.DataFrame,
    pop_sizes: list[int],
    plot_types=np.array(
        [
            "seasonality_coef",
            "vaccination_",
            "_external_introductions",
            "_strain_proportion",
            "_average_immunity",
            "total_infection_incidence",  # TODO MAKE AGE SPECIFIC
            "pred_hosp_",
        ]
    ),
    plot_titles=np.array(
        [
            "Seasonality Coefficient",
            "Vaccination Rate By Age",
            "External Introductions by Strain (per 100k)",
            "Strain Proportion of New Infections",
            "Average Population Immunity Against Strains",
            "Total Infection Incidence (per 100k)",
            "Predicted Hospitalizations (per 100k)",
        ]
    ),
    plot_normalizations=np.array([1, 1, 100000, 1, 1, 100000, 100000]),
):
    """Given a dataframe resembling the azure_visualizer_timeline csv, if it exists, returns an overview figure.
    the figure will contain 1 column per state in `timeseries_df["states"]` if the column exists. The
    figure will contain one row per

    Parameters
    ----------
    timeseries_df : _type_
        _description_
    pop_sizes : _type_
        _description_
    plot_types : _type_, optional
        _description_, by default [ "seasonality_coef", "vaccination_", "_external_introductions", "_strain_proportion", "_average_immunity", "total_infection_incidence", "pred_hosp_", ]
    plot_titles : _type_, optional
        _description_, by default [ "Seasonality Coefficient", "Vaccination Rate By Age", "External Introductions by Strain (per 100k)", "Strain Proportion of New Infections", "Average Population Immunity Against Strains", "Total Infection Incidence (per 100k)", "Predicted Hospitalizations (per 100k)", ]

    Returns
    -------
    _type_
        _description_
    """
    if "states" not in timeseries_df.columns:
        timeseries_df["states"] = "state"
    num_states = len(timeseries_df["states"].unique())
    # we are counting the number of plot_types that are within timelines.columns
    # this way we dont try to plot something that timelines does not have
    plots_in_timelines = [
        any([plot_type in col for col in timeseries_df.columns])
        for plot_type in plot_types
    ]
    num_unique_plots_in_timelines = sum(plots_in_timelines)
    # select only the plots we actually find within `timelines`
    plot_types = plot_types[plots_in_timelines].tolist()
    plot_titles = plot_titles[plots_in_timelines].tolist()
    plot_normalizations = plot_normalizations[plots_in_timelines].tolist()
    # normalize our dataframe by the given y axis normalization schemes
    timeseries_df = _cleanup_and_normalize_timelines(
        timeseries_df,
        plot_types,
        plot_normalizations,
        pop_sizes,
    )
    plt.style.use("seaborn-v0_8-colorblind")
    fig, ax = plt.subplots(
        nrows=num_unique_plots_in_timelines,
        ncols=num_states,
        sharex=True,
        sharey="row",
        squeeze=False,
        figsize=(15, 15),
    )
    # melt the df down so that each column is identified in one col rather than
    # across all cols, this makes filtering more efficient and works with seaborn style
    id_vars = ["date", "state", "chain_particle"]
    rest = [x for x in timeseries_df.columns if x not in id_vars]
    timelines_melt = pd.melt(
        timeseries_df,
        id_vars=["date", "state", "chain_particle"],
        value_vars=rest,
        var_name="column",
        value_name="val",
    )
    # convert to datetime if not already
    timelines_melt["date"] = pd.to_datetime(timelines_melt["date"])

    # go through each plot type, look for matching columns within `timelines` and plot
    # that plot_type for each chain_particle pair. plotly rows/cols are index at 1 not 0
    for state_num, state in enumerate(timeseries_df["state"].unique()):
        state_df = timelines_melt[timelines_melt["state"] == state]
        print("Plotting State : " + state)
        for plot_num, (plot_title, plot_type) in enumerate(
            zip(plot_titles, plot_types)
        ):
            plot_ax = ax[plot_num][state_num]
            # for example "vaccination_" in "vaccination_0_17" is true
            # so we include this column in the plot under that plot_type
            columns_to_plot = [
                col for col in timelines_melt.columns if plot_type in col
            ]
            df = state_df[[plot_type in x for x in state_df["column"]]]
            if len(columns_to_plot) > 1:
                df.loc[:, "column"] = df.loc[:, "column"].apply(
                    lambda x: x.replace(plot_type, "")
                )
            unique_columns = df["column"].unique()
            # plot all chain_particles as thin transparent lines
            sns.lineplot(
                df,
                x="date",
                y="val",
                hue="column",
                units="chain_particle",
                ax=plot_ax,
                estimator=None,
                alpha=0.3,
                lw=0.25,
                legend=False,
                hue_order=unique_columns,
            )
            # plot a median line of all particles with high opacity
            medians = df.groupby(by=["date", "column"])["val"].median()
            medians = medians.reset_index()
            sns.lineplot(
                medians,
                x="date",
                y="val",
                hue="column",
                ax=plot_ax,
                estimator=None,
                alpha=1.0,
                lw=2,
                legend="auto",
                hue_order=unique_columns,
            )
            # remove y labels
            plot_ax.set_ylabel("")
            plot_ax.set_title(plot_title)
            # make all legends except those on far right invisible
            plot_ax.get_legend().set_visible(False)
            # create legend for the right most plot only
            if state_num == num_states - 1:
                for lh in plot_ax.get_legend().legend_handles:
                    lh.set_alpha(1)
                plot_ax.legend(bbox_to_anchor=(1.0, 0.5), loc="center left")
    # add column titles on the top of each col for the states
    for ax, state in zip(ax[0], timeseries_df["state"].unique()):
        ax.set_title(state)
    fig.tight_layout()

    return fig
