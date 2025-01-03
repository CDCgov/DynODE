"""A series of utility functions for generating visualizations for the model"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from jax.random import PRNGKey
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap

from .utils import (
    drop_keys_with_substring,
    flatten_list_parameters,
    identify_distribution_indexes,
)


class VisualizationError(Exception):
    pass


def _cleanup_and_normalize_timeseries(
    all_state_timeseries: pd.DataFrame,
    plot_types: np.ndarray[str],
    plot_normalizations: np.ndarray[int],
    state_pop_sizes: dict[str, int],
):
    # Select columns with 'float64' dtype
    float_cols = list(all_state_timeseries.select_dtypes(include="float64"))
    # round down near-zero values to zero to make plots cleaner
    all_state_timeseries[float_cols] = all_state_timeseries[float_cols].mask(
        np.isclose(all_state_timeseries[float_cols], 0, atol=1e-4), 0
    )
    for plot_type, plot_normalization in zip(plot_types, plot_normalizations):
        for state_name, state_pop in state_pop_sizes.items():
            # if normalization is set to 1, we dont normalize at all.
            normalization_factor = (
                plot_normalization / state_pop
                if plot_normalization > 1
                else 1.0
            )
            # select all columns from that column type
            cols = [
                col for col in all_state_timeseries.columns if plot_type in col
            ]
            # update that states columns by the normalization factor
            all_state_timeseries.loc[
                all_state_timeseries["state"] == state_name,
                cols,
            ] *= normalization_factor
    return all_state_timeseries


def plot_model_overview_subplot_matplotlib(
    timeseries_df: pd.DataFrame,
    pop_sizes: dict[str, int],
    plot_types: np.ndarray[str] = np.array(
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
    plot_titles: np.ndarray[str] = np.array(
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
    plot_normalizations: np.ndarray[int] = np.array(
        [1, 1, 100000, 1, 1, 100000, 100000]
    ),
    matplotlib_style: list[str]
    | str = [
        "seaborn-v0_8-colorblind",
    ],
) -> plt.Figure:
    """Given a dataframe resembling the simulation_timeseries csv,
    if it exists, returns an overview figure. The figure will contain 1 column
    per state in `timeseries_df["state"]` if the column exists. The
    figure will contain one row per plot_type

    Parameters
    ----------
    timeseries_df : pandas.DataFrame
        a dataframe containing at least the following columns:
        ["date", "chain_particle", "state"] followed by columns identifying
        different timeseries of interest to be plotted.
        E.g. vaccination_0_17, vaccination_18_49, total_infection_incidence.
        columns that share the same plot_type will be plotted on the same plot,
        with their differences in the legend.
        All chain_particle replicates are plotted as low
        opacity lines for each plot_type
    pop_sizes : dict[str, int]
        population sizes of each state as a dictionary.
        Keys must match the "state" column within timeseries_df
    plot_types : np.ndarray[str], optional
        each of the plot types to be plotted.
        plot_types not found in `timeseries_df` are skipped.
        columns are identified using the "in" operation,
        so plot_type must be found in each of its identified columns
        by default ["seasonality_coef", "vaccination_",
        "_external_introductions", "_strain_proportion", "_average_immunity",
        "total_infection_incidence", "pred_hosp_"]
    plot_titles : np.ndarray[str], optional
        titles for each plot_type as displayed on each subplot,
        by default [ "Seasonality Coefficient", "Vaccination Rate By Age",
        "External Introductions by Strain (per 100k)",
        "Strain Proportion of New Infections",
        "Average Population Immunity Against Strains",
        "Total Infection Incidence (per 100k)",
        "Predicted Hospitalizations (per 100k)"]
    plot_normalizations : np.ndarray[int]
        normalization factor for each plot type
    matplotlib_style: list[str] | str
        matplotlib style to plot in, by default ["seaborn-v0_8-colorblind"]

    Returns
    -------
    matplotlib.pyplot.Figure
        matplotlib Figure containing subplots with a column for each state
        and a row for each plot_type
    """
    necessary_cols = ["date", "chain_particle", "state"]
    assert all(
        [
            necessary_col in timeseries_df.columns
            for necessary_col in necessary_cols
        ]
    ), (
        "missing a necessary column within timeseries_df, require %s but got %s"
        % (str(necessary_cols), str(timeseries_df.columns))
    )
    num_states = len(timeseries_df["state"].unique())
    # we are counting the number of plot_types that are within timeseries.columns
    # this way we dont try to plot something that timeseries does not have
    plots_in_timeseries = [
        any([plot_type in col for col in timeseries_df.columns])
        for plot_type in plot_types
    ]
    num_unique_plots_in_timeseries = sum(plots_in_timeseries)
    # select only the plots we actually find within `timeseries`
    plot_types = plot_types[plots_in_timeseries]
    plot_titles = plot_titles[plots_in_timeseries]
    plot_normalizations = plot_normalizations[plots_in_timeseries]
    # normalize our dataframe by the given y axis normalization schemes
    timeseries_df = _cleanup_and_normalize_timeseries(
        timeseries_df,
        plot_types,
        plot_normalizations,
        pop_sizes,
    )
    with plt.style.context(matplotlib_style):
        fig, ax = plt.subplots(
            nrows=num_unique_plots_in_timeseries,
            ncols=num_states,
            sharex=True,
            sharey="row",
            squeeze=False,
            figsize=(6 * num_states, 3 * num_unique_plots_in_timeseries),
        )
    # melt this df down to have an ID column "column" and a value column "val"
    id_vars = ["date", "state", "chain_particle"]
    rest = [x for x in timeseries_df.columns if x not in id_vars]
    timeseries_melt = pd.melt(
        timeseries_df,
        id_vars=["date", "state", "chain_particle"],
        value_vars=rest,
        var_name="column",
        value_name="val",
    )
    # convert to datetime if not already
    timeseries_melt["date"] = pd.to_datetime(timeseries_melt["date"])

    # go through each plot type, look for matching columns and plot
    # that plot_type for each chain_particle pair.
    for state_num, state in enumerate(timeseries_df["state"].unique()):
        state_df = timeseries_melt[timeseries_melt["state"] == state]
        print("Plotting State : " + state)
        for plot_num, (plot_title, plot_type) in enumerate(
            zip(plot_titles, plot_types)
        ):
            plot_ax = ax[plot_num][state_num]
            # for example "vaccination_" in "vaccination_0_17" is true
            # so we include this column in the plot under that plot_type
            plot_df = state_df[[plot_type in x for x in state_df["column"]]]
            columns_to_plot = plot_df["column"].unique()
            # if we are plotting multiple lines, lets modify the legend to
            # only display the differences between each line
            if len(columns_to_plot) > 1:
                plot_df.loc[:, "column"] = plot_df.loc[:, "column"].apply(
                    lambda x: x.replace(plot_type, "")
                )
            unique_columns = plot_df["column"].unique()
            # plot all chain_particles as thin transparent lines
            # turn off legends since there will many lines
            sns.lineplot(
                plot_df,
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
            # use this as our legend line
            medians = (
                plot_df.groupby(by=["date", "column"])["val"]
                .median()
                .reset_index()
            )
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
                with plt.style.context(matplotlib_style):
                    for lh in plot_ax.get_legend().legend_handles:
                        lh.set_alpha(1)
                    plot_ax.legend(
                        bbox_to_anchor=(1.0, 0.5),
                        loc="center left",
                    )
    # add column titles on the top of each col for the states
    for ax, state in zip(ax[0], timeseries_df["state"].unique()):
        plot_title = ax.get_title()
        ax.set_title(plot_title + "\n" + state)
    fig.tight_layout()

    return fig


def plot_checkpoint_inference_correlation_pairs(
    posteriors: dict[str : np.ndarray | list],
    max_samples_calculated: int = 100,
    matplotlib_style: list[str]
    | str = [
        "seaborn-v0_8-colorblind",
    ],
):
    """Given a dictionary mapping a sampled parameter's name to its
    posteriors samples, returns a figure plotting
    the correlation of each sampled parameter with all other sampled parameters
    on the upper half of the plot the correlation values, on the diagonal a
    historgram of the posterior values, and on the bottom half a scatter
    plot of the parameters against eachother along with a matching trend line.


    Parameters
    ----------
    posteriors: dict[str : np.ndarray | list]
        a dictionary (usually loaded from the checkpoint.json file) containing
        the sampled posteriors for each chain in the shape
        (num_chains, num_samples). All parameters generated with numpyro.plate
        and thus have a third dimension (num_chains, num_samples, num_plates)
        are flattened to the desired shape and displayed as
        separate parameters with _i suffix for each i in num_plates.
    max_samples_calculated: int
        a max cap of posterior samples per chain on which
        calculations such as correlations and plotting will be performed
        set for efficiency of plot generation,
        set to -1 to disable cap, by default 100
    matplotlib_style: list[str] | str
        matplotlib style to plot in, by default ["seaborn-v0_8-colorblind"]

    Returns
    -------
    matplotlib.pyplot.Figure
        Figure with `n` rows and `n` columns where
        `n` is the number of sampled parameters
    """
    # convert lists to np.arrays
    posteriors = {
        key: np.array(val) if isinstance(val, list) else val
        for key, val in posteriors.items()
    }
    posteriors: dict[str, np.ndarray] = flatten_list_parameters(posteriors)
    # drop any timestep parameters in case they snuck in
    posteriors = drop_keys_with_substring(posteriors, "timestep")
    number_of_samples = posteriors[list(posteriors.keys())[0]].shape[1]
    # if we are dealing with many samples per chain,
    # narrow down to max_samples_calculated samples per chain
    if (
        number_of_samples > max_samples_calculated
        and max_samples_calculated != -1
    ):
        selected_indices = np.random.choice(
            number_of_samples, size=max_samples_calculated, replace=False
        )
        posteriors = {
            key: matrix[:, selected_indices]
            for key, matrix in posteriors.items()
        }
    number_of_samples = posteriors[list(posteriors.keys())[0]].shape[1]
    # Flatten matrices including chains and create Correlation DataFrame
    posteriors = {
        key: np.array(matrix).flatten() for key, matrix in posteriors.items()
    }
    columns = posteriors.keys()
    num_cols = len(list(columns))
    label_size = max(2, min(10, 200 / num_cols))
    # Compute the correlation matrix, reverse it so diagonal starts @ top left
    samples_df = pd.DataFrame(posteriors)
    correlation_df = samples_df.corr(method="pearson")

    cmap = LinearSegmentedColormap.from_list("", ["red", "grey", "blue"])

    def _normalize_coefficients_to_0_1(r):
        # squashes [-1, 1] into [0, 1] via (r - min()) / (max() - min())
        return (r + 1) / 2

    def reg_coef(x, y, label=None, color=None, **kwargs):
        ax = plt.gca()
        x_name, y_name = (x.name, y.name)
        r = correlation_df.loc[x_name, y_name]
        ax.annotate(
            "{:.2f}".format(r),
            xy=(0.5, 0.5),
            xycoords="axes fraction",
            ha="center",
            # vary size and color by the magnitude of correlation
            color=cmap(_normalize_coefficients_to_0_1(r)),
            size=label_size * abs(r) + label_size,
        )
        ax.set_axis_off()

    def reg_plot_custom(x, y, label=None, color=None, **kwargs):
        ax = plt.gca()
        x_name, y_name = (x.name, y.name)
        r = correlation_df.loc[x_name, y_name]
        ax = sns.regplot(
            x=x,
            y=y,
            ax=ax,
            fit_reg=True,
            scatter_kws={"alpha": 0.2, "s": 0.5},
            line_kws={
                "color": cmap(_normalize_coefficients_to_0_1(r)),
                "linewidth": 1,
            },
        )

    # Create the plot
    with plt.style.context(matplotlib_style):
        g = sns.PairGrid(
            data=samples_df,
            vars=columns,
            diag_sharey=False,
            layout_pad=0.01,
        )
    g.map_upper(reg_coef)
    g = g.map_lower(
        reg_plot_custom,
    )
    g = g.map_diag(sns.histplot, kde=True)
    for ax in g.axes.flatten():
        plt.setp(ax.get_xticklabels(), rotation=45, size=label_size)
        plt.setp(ax.get_yticklabels(), rotation=45, size=label_size)
        # extract the existing xaxis label
        xlabel = ax.get_xlabel()
        # set the xaxis label with rotation
        ax.set_xlabel(xlabel, size=label_size, rotation=90, labelpad=4.0)

        ylabel = ax.get_ylabel()
        ax.set_ylabel(ylabel, size=label_size, rotation=0, labelpad=15.0)
        ax.label_outer(remove_inner_ticks=True)
    # Adjust layout to make sure everything fits
    px = 1 / plt.rcParams["figure.dpi"]
    g.figure.set_size_inches((2000 * px, 2000 * px))
    # g.figure.tight_layout(pad=0.01, h_pad=0.01, w_pad=0.01)
    return g.figure


def plot_mcmc_chains(
    samples: dict[str : np.ndarray | list],
    matplotlib_style: list[str]
    | str = [
        "seaborn-v0_8-colorblind",
    ],
) -> plt.Figure:
    """given a `samples` dictionary containing posterior samples
    often returned from numpyro.get_samples(group_by_chain=True)
    or from the checkpoint.json saved file, plots each MCMC chain
    for each sampled parameter in a roughly square subplot.

    Parameters
    ----------
    posteriors: dict[str : np.ndarray | list]
        a dictionary (usually loaded from the checkpoint.json file) containing
        the sampled posteriors for each chain in the shape
        (num_chains, num_samples). All parameters generated with numpyro.plate
        and thus have a third dimension (num_chains, num_samples, num_plates)
        are flattened to the desired and displayed as
        separate parameters with _i suffix for each i in num_plates.
    matplotlib_style : list[str] | str, optional
        matplotlib style to plot in by default ["seaborn-v0_8-colorblind"]

    Returns
    -------
    matplotlib.pyplot.Figure
        matplotlib figure containing the plots
    """
    # Determine the number of parameters and chains
    samples = {
        key: np.array(val) if isinstance(val, list) else val
        for key, val in samples.items()
    }
    samples: dict[str, np.ndarray] = flatten_list_parameters(samples)
    # drop any timestep parameters in case they snuck in
    samples = drop_keys_with_substring(samples, "timestep")
    param_names = list(samples.keys())
    num_params = len(param_names)
    num_chains = samples[param_names[0]].shape[0]
    # Calculate the number of rows and columns for a square-ish layout
    num_cols = int(np.ceil(np.sqrt(num_params)))
    num_rows = int(np.ceil(num_params / num_cols))
    # Create a figure with subplots
    with plt.style.context(matplotlib_style):
        fig, axs = plt.subplots(
            num_rows,
            num_cols,
            figsize=(3 * num_cols, 3 * num_rows),
            squeeze=False,
        )
    # Flatten the axis array for easy indexing
    axs_flat = axs.flatten()
    # Loop over each parameter and plot its chains
    for i, param_name in enumerate(param_names):
        ax: Axes = axs_flat[i]
        for chain in range(num_chains):
            ax.plot(samples[param_name][chain], label=f"chain {chain}")
        ax.set_title(param_name)
        # Hide x-axis labels except for bottom plots to reduce clutter
        if i < (num_params - num_cols):
            ax.set_xticklabels([])

    # Turn off any unused subplots
    for j in range(i + 1, len(axs_flat)):
        axs_flat[j].axis("off")
    plt.tight_layout()
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center")
    return fig


def plot_prior_distributions(
    priors: dict[str],
    matplotlib_style: list[str]
    | str = [
        "seaborn-v0_8-colorblind",
    ],
    num_samples=5000,
    hist_kwargs={"bins": 50, "density": True},
) -> plt.Figure:
    """Given a dictionary of parameter keys and possibly values of
    numpyro.distribution objects, samples them a number of times
    and returns a plot of those samples to help
    visualize the range of values taken by that prior distribution.

    Parameters
    ----------
    priors : dict[str: Any]
        a dictionary with str keys possibly containing distribution
        objects as values. Each key with a distribution object type
        key will be included in the plot
    matplotlib_style : list[str] | str, optional
        matplotlib style to plot in by default ["seaborn-v0_8-colorblind"]
    num_samples: int, optional
        the number of times to sample each distribution, mild impact on
        figure performance. By default 50000
    hist_kwargs: dict[str: Any]
        additional kwargs passed to plt.hist(), by default {"bins": 50}

    Returns
    -------
    plt.Figure
        matplotlib figure that is roughly square containing all distribution
        keys found within priors.
    """
    dist_only = {}
    d = identify_distribution_indexes(priors)
    # filter down to just the distribution objects
    for dist_name, locator_dct in d.items():
        parameter_name = locator_dct["sample_name"]
        parameter_idx = locator_dct["sample_idx"]
        # if the sample is on its own, not nested in a list, sample_idx is none
        if parameter_idx is None:
            dist_only[parameter_name] = priors[parameter_name]
        # otherwise this sample is nested in a list and should be retrieved
        else:
            # go in index by index to access multi-dimensional lists
            temp = priors[parameter_name]
            for i in parameter_idx:
                temp = temp[i]
            dist_only[dist_name] = temp
    param_names = list(dist_only.keys())
    num_params = len(param_names)
    if num_params == 0:
        raise VisualizationError(
            "Attempted to visualize a config without any distributions"
        )
    # Calculate the number of rows and columns for a square-ish layout
    num_cols = int(np.ceil(np.sqrt(num_params)))
    num_rows = int(np.ceil(num_params / num_cols))
    with plt.style.context(matplotlib_style):
        fig, axs = plt.subplots(
            num_rows,
            num_cols,
            figsize=(3 * num_cols, 3 * num_rows),
            squeeze=False,
        )
    # Flatten the axis array for easy indexing
    axs_flat = axs.flatten()
    # Loop over each parameter and sample
    for i, param_name in enumerate(param_names):
        ax: Axes = axs_flat[i]
        ax.set_title(param_name)
        dist = dist_only[param_name]
        samples = dist.sample(PRNGKey(0), sample_shape=(num_samples,))
        ax.hist(samples, **hist_kwargs)
        ax.axvline(
            samples.mean(),
            linestyle="dashed",
            linewidth=1,
            label="mean",
        )
        ax.axvline(
            jnp.median(samples),
            linestyle="dotted",
            linewidth=3,
            label="median",
        )
    # Turn off any unused subplots
    for j in range(i + 1, len(axs_flat)):
        axs_flat[j].axis("off")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper right")
    fig.suptitle("Prior Distributions Visualized, n=%s" % num_samples)
    plt.tight_layout()
    return fig
