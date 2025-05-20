"""A set of utility functions for generating visualizations for the model."""

from typing import Any

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from jax import Array
from jax.random import PRNGKey
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap

from . import (
    drop_keys_with_substring,
    flatten_list_parameters,
    identify_distribution_indexes,
)


class VisualizationError(Exception):
    """An exception class for Visualization Errors."""

    pass


def _cleanup_and_normalize_timeseries(
    all_region_timeseries: pd.DataFrame,
    plot_types: np.ndarray,
    plot_normalizations: np.ndarray,
    region_pop_sizes: dict[str, int],
):
    # Select columns with 'float64' dtype
    float_cols = list(all_region_timeseries.select_dtypes(include="float64"))
    # round down near-zero values to zero to make plots cleaner
    all_region_timeseries[float_cols] = all_region_timeseries[float_cols].mask(
        np.isclose(all_region_timeseries[float_cols], 0, atol=1e-4), 0
    )
    for plot_type, plot_normalization in zip(plot_types, plot_normalizations):
        for region_name, region_pop in region_pop_sizes.items():
            # if normalization is set to 1, we dont normalize at all.
            normalization_factor = (
                plot_normalization / region_pop
                if plot_normalization > 1
                else 1.0
            )
            # select all columns from that column type
            cols = [
                col
                for col in all_region_timeseries.columns
                if plot_type in col
            ]
            # update that region columns by the normalization factor
            all_region_timeseries.loc[
                all_region_timeseries["region"] == region_name,
                cols,
            ] *= normalization_factor
    return all_region_timeseries


def plot_model_overview_subplot_matplotlib(
    timeseries_df: pd.DataFrame,
    pop_sizes: dict[str, int],
    plot_types: np.ndarray = np.array(
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
    plot_titles: np.ndarray = np.array(
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
    plot_normalizations: np.ndarray = np.array(
        [1, 1, 100000, 1, 1, 100000, 100000]
    ),
    matplotlib_style: list[str] | str = [
        "seaborn-v0_8-colorblind",
    ],
) -> plt.Figure:
    """Generate an overview figure containing subplots for various model metrics.

    Parameters
    ----------
    timeseries_df : pd.DataFrame
        DataFrame containing at least ["date", "chain_particle", "region"]
        followed by columns for different time series to be plotted.

    pop_sizes : dict[str, int]
        Population sizes for each region as a dictionary. Keys must match
        the values in the "region" column of `timeseries_df`.

    plot_types : np.ndarray[str], optional
        Types of plots to be generated.
        Elements not found in `timeseries_df` are skipped.

    plot_titles : np.ndarray[str], optional
        Titles for each subplot corresponding to `plot_types`.

    plot_normalizations : np.ndarray[int]
        Normalization factors for each plot type.

    matplotlib_style: list[str] | str
        Matplotlib style to use for plotting.

    Returns
    -------
    plt.Figure
        Matplotlib Figure containing subplots with one column per region
        and one row per plot type.
    """
    necessary_cols = ["date", "chain_particle", "region"]
    assert all(
        [
            necessary_col in timeseries_df.columns
            for necessary_col in necessary_cols
        ]
    ), (
        "missing a necessary column within timeseries_df, require %s but got %s"
        % (
            str(necessary_cols),
            str(timeseries_df.columns),
        )
    )
    num_regions = len(timeseries_df["region"].unique())
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
            ncols=num_regions,
            sharex=True,
            sharey="row",
            squeeze=False,
            figsize=(6 * num_regions, 3 * num_unique_plots_in_timeseries),
        )
    # melt this df down to have an ID column "column" and a value column "val"
    id_vars = ["date", "region", "chain_particle"]
    rest = [x for x in timeseries_df.columns if x not in id_vars]
    timeseries_melt = pd.melt(
        timeseries_df,
        id_vars=["date", "region", "chain_particle"],
        value_vars=rest,
        var_name="column",
        value_name="val",
    )
    # convert to datetime if not already
    timeseries_melt["date"] = pd.to_datetime(timeseries_melt["date"])

    # go through each plot type, look for matching columns and plot
    # that plot_type for each chain_particle pair.
    for region_num, region in enumerate(timeseries_df["region"].unique()):
        region_df = timeseries_melt[timeseries_melt["region"] == region]
        print("Plotting Region : " + region)
        for plot_num, (plot_title, plot_type) in enumerate(
            zip(plot_titles, plot_types)
        ):
            plot_ax = ax[plot_num][region_num]
            # for example "vaccination_" in "vaccination_0_17" is true
            # so we include this column in the plot under that plot_type
            plot_df = region_df[[plot_type in x for x in region_df["column"]]]
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
            if region_num == num_regions - 1:
                with plt.style.context(matplotlib_style):
                    for lh in plot_ax.get_legend().legend_handles:
                        lh.set_alpha(1)
                    plot_ax.legend(
                        bbox_to_anchor=(1.0, 0.5),
                        loc="center left",
                    )
    # add column titles on the top of each col for the regions
    for ax, region in zip(ax[0], timeseries_df["region"].unique()):
        plot_title = ax.get_title()
        ax.set_title(plot_title + "\n" + region)
    fig.tight_layout()

    return fig


def plot_checkpoint_inference_correlation_pairs(
    posteriors_in: dict[str, np.ndarray | list],
    max_samples_calculated: int = 100,
    matplotlib_style: list[str] | str = [
        "seaborn-v0_8-colorblind",
    ],
):
    """Plot correlation pairs of sampled parameters with histograms and trend lines.

    Parameters
    ----------
    posteriors_in : dict[str, np.ndarray | list]
        Dictionary mapping parameter names to their posterior samples
        (shape: num_chains, num_samples). Parameters generated with
        numpyro.plate are flattened and displayed as separate parameters
        with _i suffix for each i in num_plates.

    max_samples_calculated : int
        Maximum number of posterior samples per chain for calculations
        such as correlations and plotting. Set to -1 to disable cap; default is 100.

    matplotlib_style : list[str] | str
        Matplotlib style to use for plotting; default is ["seaborn-v0_8-colorblind"].

    Returns
    -------
    plt.Figure
        Figure with n rows and n columns where n is the number of sampled parameters.
    """
    # convert lists to np.arrays
    posteriors: dict[str, np.ndarray | Array] = flatten_list_parameters(
        {
            key: np.array(val) if isinstance(val, list) else val
            for key, val in posteriors_in.items()
        }
    )
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
    samples_in: dict[str, np.ndarray | list],
    matplotlib_style: list[str] | str = [
        "seaborn-v0_8-colorblind",
    ],
) -> plt.Figure:
    """Plot MCMC chains for each sampled parameter in a grid of subplots.

    Parameters
    ----------
    samples_in : dict[str, np.ndarray | list]
        Dictionary containing posterior samples (shape: num_chains, num_samples).
        Parameters generated with numpyro.plate are flattened and displayed as
        separate parameters with _i suffix for each i in num_plates.

    matplotlib_style : list[str] | str, optional
        Matplotlib style to use for plotting;
        default is ["seaborn-v0_8-colorblind"].

    Returns
    -------
    plt.Figure
        Matplotlib figure containing the plots.
    """
    # Determine the number of parameters and chains
    samples: dict[str, np.ndarray | Array] = flatten_list_parameters(
        {
            key: np.array(val) if isinstance(val, list) else val
            for key, val in samples_in.items()
        }
    )
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


def _sample_prior_distributions(priors, num_samples) -> dict[str, Array]:
    """Sample numpyro.distributions `num_samples` times.

    Parameters
    ----------
    priors : dict[str, Any]
        A dictionary containing keys of different parameter
        names and values of any type.
    num_samples : int
        number of times to sample numpyro.distribution objects.

    Returns
    -------
    dict[str, jax.Array]
        Numpyro sample site name with jax.Array(shape=(num_samples,)) for each
        numpyro.distribution found within `priors`.

    Notes
    -----
    Return dict key names follow the same naming convention as when sampling.
    Meaning that distributions within lists or matricies have their
    index stored as a list of _i suffix at the end of their name.
    """
    dist_only = {}
    d = identify_distribution_indexes(priors)
    # filter down to just the distribution objects
    for dist_name, locator_dct in d.items():
        parameter_name = locator_dct["sample_name"]
        assert isinstance(parameter_name, str)
        parameter_idx = locator_dct["sample_idx"]
        assert isinstance(parameter_idx, tuple) or parameter_idx is None
        # if the sample is on its own, not nested in a list, sample_idx is none
        if parameter_idx is None:
            dist_only[parameter_name] = priors[parameter_name]
        # otherwise this sample is nested in a list and should be retrieved
        else:
            temp = priors[parameter_name]
            # go into multi-dimensional matricies one index at a time
            for i in parameter_idx:
                temp = temp[i]
            dist_only[dist_name] = temp
    sampled_priors = {}
    for param, dist in dist_only.items():
        sampled_priors[param] = dist.sample(
            PRNGKey(0), sample_shape=(num_samples,)
        )
    return sampled_priors


def plot_prior_distributions(
    priors: dict[str, Any],
    matplotlib_style: list[str] | str = [
        "seaborn-v0_8-colorblind",
    ],
    num_samples=5000,
    hist_kwargs={"bins": 50, "density": True},
    median_line_kwargs={
        "linestyle": "dotted",
        "linewidth": 3,
        "label": "prior median",
    },
) -> plt.Figure:
    """Visualize prior distributions by sampling from them and plotting the results.

    Parameters
    ----------
    priors : dict[str, Any]
        Dictionary with string keys and distribution
        objects as values. Each key with a distribution object will be
        included in the plot.

    matplotlib_style : list[str] | str, optional
        Matplotlib style to use for plotting;
        default is ["seaborn-v0_8-colorblind"].

    num_samples : int, optional
        Number of times to sample each distribution;
        default is 5000.

    hist_kwargs : dict[str: Any]
        Additional kwargs passed to `plt.hist()`; default is {"bins": 50}.

    Returns
    -------
    plt.Figure
        Matplotlib figure containing all distribution keys found within `priors`.
    """
    sampled_priors = _sample_prior_distributions(priors, num_samples)
    param_names = list(sampled_priors.keys())
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
        samples = sampled_priors[param_name]
        ax.hist(samples, **hist_kwargs)
        ax.axvline(float(jnp.median(samples)), **median_line_kwargs)
        # testing
    # Turn off any unused subplots
    for j in range(i + 1, len(axs_flat)):
        axs_flat[j].axis("off")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper right")
    fig.suptitle("Prior Distributions Visualized, n=%s" % num_samples)
    plt.tight_layout()
    return fig


def plot_violin_plots(
    priors: dict[str, list] | None = None,
    posteriors: dict[str, list] | None = None,
    matplotlib_style: list[str] | str = [
        "seaborn-v0_8-colorblind",
    ],
):
    """Save violin plot of priors and posteriors together.

    Parameters
    ----------
    priors : dict[str, list], optional
        samples from a parameter's prior distribution, by default None
    posteriors : dict[str, list], optional
        samples from a parameter's posterior distribution, by default None
    matplotlib_style : list[str] | str, optional
        matplotlib style(s) to apply, by default [ "seaborn-v0_8-colorblind",]

    Returns
    -------
    matplotlib.Figure
        matplotlib Figure containing violin plots of the priors and posteriors.

    Raises
    ------
    VisualizationError
        if both `priors` and `posteriors` is None there is nothing to plot.

    Notes
    -----
    Returned figure will be roughly square containing N subplots for N
    parameters.

    If some parameters are missing in either dictionary, an open space in
    that subplot will be left in the figure.
    """
    if priors is None and posteriors is None:
        raise VisualizationError(
            "must provide either a dictionary of priors or posteriors"
        )
    # we are given that both are not none, so get num_params from one of them
    if posteriors is not None:
        num_params = len(posteriors.keys())
    elif priors is not None:
        num_params = len(priors.keys())
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
        axs = axs.flatten()

    df = pd.DataFrame()
    if priors is not None:
        for param, values in priors.items():
            df_param = pd.DataFrame()
            # flatten any chains if they leaked in
            df_param["values"] = np.array(values).flatten()
            df_param["type"] = "prior"
            df_param["param"] = param
            df = pd.concat([df, df_param], ignore_index=True, axis=0)
    if posteriors is not None:
        for param, values in posteriors.items():
            df_param = pd.DataFrame()
            # flatten any chains if they leaked in
            df_param["values"] = np.array(values).flatten()
            df_param["type"] = "posterior"
            df_param["param"] = param
            # this is necessary to make sure there are always two columns
            # of violin plots, including when a posterior does not have an
            # associated prior
            if priors is not None and param not in priors.keys():
                filler_row = pd.DataFrame(
                    {"values": [np.nan], "type": "prior", "param": param}
                )
                df_param = pd.concat(
                    [filler_row, df_param], ignore_index=True, axis=0
                )
            df = pd.concat([df, df_param], ignore_index=True, axis=0)

    # parameters that share a first word will be colored the same for interpretability
    unique_first_words = set(
        [param.split("_")[0] for param in df["param"].unique()]
    )
    color_palette = sns.color_palette("Set2", n_colors=len(unique_first_words))
    color_dict = dict(zip(unique_first_words, color_palette))

    # Iterate over the parameters and create violin plots
    for i, param in enumerate(df["param"].unique()):
        ax: Axes = axs[i]
        # if priors is not None and param in priors.keys():
        # sns.violinplot(y=priors[param], ax=ax, alpha=0.5, label="prior")
        sns.violinplot(
            data=df.loc[df["param"] == param],
            x="type",
            y="values",
            ax=ax,
            color=color_dict[param.split("_")[0]],
        )
        ax.set_title(param)
        ax.set_ylabel("")
        ax.set_xlabel("")

    # Remove empty subplots if necessary
    if num_params < num_rows * num_cols:
        for i in range(num_params, num_rows * num_cols):
            fig.delaxes(axs[i])
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper right")
    fig.suptitle("Violin Plot of Parameters")
    fig.tight_layout()
    return fig
