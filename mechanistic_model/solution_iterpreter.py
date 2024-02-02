"""
The following class interprets Solution results into digestible figures 
and is responsible for ensuring reproducibility and replicability of model outputs.
"""
from diffrax import Solution
import matplotlib.pyplot as plt
from datetime import datetime
import json
import numpy as np
import jax.numpy as jnp
from enum import EnumMeta
from config.config_parser import ConfigParser
import utils


class SolutionInterpreter:
    def __init__(self, solution, solution_parameters, global_variables):
        if isinstance(solution_parameters, str):
            solution_parameters = ConfigParser(
                solution_parameters
            ).get_config()

        if isinstance(global_variables, str):
            global_variables = ConfigParser(global_variables).get_config()

        self.__dict__.update(global_variables)
        self.__dict__.update(solution_parameters)
        self.solution = solution
        self.solution_parameters = solution_parameters
        self.pyplot_theme = None  # TODO set a consistent theme

    def set_default_plot_commands(self, plot_commands):
        """
        Where applicable, the solution interpreter will plot the given plot_commands by default.
        """
        self.PLOT_COMMANDS = plot_commands

    def summarize_solution(
        self,
        plot_commands: list[str] = ["S", "E", "I", "C"],
        plot_labels: list[str] = ["S", "E", "I", "C"],
        save_path: str = None,
    ):
        fig, axs = plt.subplots(2, 2, figsize=(8, 9))
        # plot commands with unlogged y axis
        fig, axs[0][0] = self.plot_solution(
            self.solution.ys,
            plot_commands,
            plot_labels,
            log_scale=False,
            fig=fig,
            ax=axs[0][0],
        )
        # plot commands with logged y axis
        fig, axs[1][0] = self.plot_solution(
            self.solution.ys,
            plot_commands,
            plot_labels,
            log_scale=True,
            fig=fig,
            ax=axs[1][0],
        )
        # strain prevalence chart over the same x axis, no plot commands.
        fig, axs[0][1] = self.plot_strain_prevalence(
            self.solution.ys, fig=fig, ax=axs[0][1]
        )
        # incidence scatter plot, unlogged y axis
        fig, axs[1][1] = self.plot_diffrax_solution(
            self.solution.ys,
            ["incidence"],
            log_scale=False,
            fig=fig,
            ax=axs[1][1],
        )

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        # save if user passes str save_path
        self.save_plot(save_path, fig)
        return fig, axs

    def plot_solution(
        self,
        plot_commands: list[str] = ["S", "E", "I", "C"],
        plot_labels: list[str] = None,
        save_path: str = None,
        log_scale: bool = None,
        start_date: datetime.date = None,
        fig: plt.figure = None,
        ax: plt.axis = None,
    ):
        """
        plots a run from diffeqsolve() with `plot_commands` returning figure and axis.
        If `save_path` is not None will save figure to that path attached with meta data in `meta_data`.

        if log_scale is unspecified, plots both logged and unlogged y axis of the `plot_commands` supplied.

        Parameters
        ----------
        sol : difrax.Solution
            object containing ODE run as described by https://docs.kidger.site/diffrax/api/solution/
        plot_commands : list(str), optional
            commands to the plotter on which populations to show, may be compartment titles, strain names, waning compartments, or explicit numpy slices!
            see utils/get_timeline_from_solution_with_command() for more in depth explanation of commands.
        plot_labels: list(str), optional
            labels to give to the lines plotted by each plot_command. These will appear in the legend of the figure.
            If not specified will default behavior to plot commands themselves.
        save_path : str, optional
            if `save_path = None` do not save figure to output directory. Otherwise save to relative path `save_path`
            attaching meta data of the self object.
        log_scale : bool, optional
            whether or not to exclusively show the log or unlogged version of the plot, by default include both
            in a stacked subplot.
        start_date : date, optional
            the start date of the x axis of the plot. Defaults to model.INIT_DATE + model.DAYS_AFTER_INIT_DATE
        fig: matplotlib.pyplot.figure
            if this plot is part of a larger subplots, pass the figure object here, otherwise one is created
        ax: matplotlib.pyplot.axis
            if this plot is part of a larger subplots, pass the specific axis object here, otherwise one is created

        Returns
        ----------
        fig, ax : matplotlib.Figure/axis object
            objects containing the matplotlib figure and axis for further modifications if needed.
        """
        # default start date is based on the model INIT date and in the case of epochs, days after initialization
        if start_date is None:
            start_date = self.INIT_DATE
        plot_commands = [x.strip() for x in plot_commands]
        if fig is None or ax is None:
            fig, ax = plt.subplots(
                2 if log_scale is None else 1, figsize=(8, 9)
            )
            # plotting both logged and unlogged is recursive calls
            if log_scale is None:
                fig, ax[0] = self.plot_diffrax_solution(
                    sol,
                    plot_commands,
                    log_scale=False,
                    fig=fig,
                    ax=ax[0],
                )
                fig, ax[1] = self.plot_diffrax_solution(
                    sol,
                    plot_commands,
                    log_scale=True,
                    fig=fig,
                    ax=ax[1],
                )
                # clear plot commands since everything was done recursively
                plot_commands = []
        sol = sol.ys
        for idx, command in enumerate(plot_commands):
            timeline, label = utils.get_timeline_from_solution_with_command(
                sol,
                self.COMPARTMENT_IDX,
                self.W_IDX,
                self.STRAIN_IDX,
                command,
            )
            # if we explicitly set plot_labels, override the default ones.
            label = plot_labels[idx] if plot_labels is not None else label
            days = list(range(len(timeline)))
            x_axis = [
                start_date + datetime.timedelta(days=day) for day in days
            ]
            if command == "incidence":
                # plot both logged and unlogged version by default
                timeline = np.log(timeline) if log_scale else timeline
                ax.scatter(x_axis, timeline, label=label, s=1)
            else:  # all other commands plot lines
                ax.plot(
                    x_axis,
                    timeline,
                    label=label,
                )
        # plotting both plots, dont want to have text overlap.
        if log_scale is None:
            fig.tight_layout()
        # dont want to set title again if we did it recursively above
        if log_scale is not None:
            ax.tick_params(axis="x", labelrotation=45)
            ax.legend()
            ax.set_ylabel("Population Count", fontsize=8)
        if log_scale:  # if single plot, user wants logged or not
            ax.set_yscale("log")
        # save if user passes str save_path
        self.save_plot(save_path, fig)
        return fig, ax

    def plot_strain_prevalence(
        self,
        sol: Solution,
        plot_labels=None,
        save_path: str = None,
        fig: plt.figure = None,
        ax: plt.axis = None,
    ):
        """
        Function that plots only strain prevalence by day. Follows similar schema to `plot_diffrax_solution()` but takes no plot commands.
        If `save_path` is not None will save figure to that path attached with meta data in `meta_data`.

        Parameters
        ----------
        sol : difrax.Solution
            object containing ODE run as described by https://docs.kidger.site/diffrax/api/solution/
        plot_labels: list(str), optional
            labels to give to the lines plotted by each strain. These will appear in the legend of the figure.
            If not specified will default behavior to the strain names.
        save_path : str, optional
            if `save_path = None` do not save figure to output directory. Otherwise save to relative path `save_path`
            attaching meta data of the self object.
        fig: matplotlib.pyplot.figure
            if this plot is part of a larger subplots, pass the figure object here, otherwise one is created
        ax: matplotlib.pyplot.axis
            if this plot is part of a larger subplots, pass the specific axis object here, otherwise one is created

        Returns
        ----------
        fig, ax : matplotlib.Figure/axis object
            objects containing the matplotlib figure and axis for further modifications if needed.
        """
        if fig is None or ax is None:
            fig, ax = plt.subplots(1, figsize=(8, 9))
        sol = sol.ys
        (
            strain_prevalence_arr,
            labels,
        ) = utils.get_timeline_from_solution_with_command(
            sol,
            self.COMPARTMENT_IDX,
            self.W_IDX,
            self.STRAIN_IDX,
            "strain_prevalence",
        )
        plot_labels = labels if plot_labels is None else plot_labels
        # create x axis by using the number of days in the array
        days = list(range(len(strain_prevalence_arr[0])))
        x_axis = [
            self.INIT_DATE + datetime.timedelta(days=day) for day in days
        ]
        ax.stackplot(
            x_axis, *strain_prevalence_arr, labels=plot_labels, baseline="zero"
        )
        ax.tick_params(axis="x", labelrotation=45)
        ax.legend()
        ax.set_ylabel("Proportion of infected population", fontsize=8)
        self.save_plot(save_path, fig)
        return fig, ax

    def plot_initial_serology(
        self, save_path: str = None, show: bool = True, fig=None, ax=None
    ):
        """
        plots a stacked bar chart representation of the initial immune compartments of the model.

        Parameters
        ----------
        save_path: {str, None}, optional
            the save path to which to save the figure, None implies figure will not be saved.
        show: {Boolean, None}, optional
            Whether or not to show the figure using plt.show() defaults to True.
        fig: matplotlib.pyplot.figure
            if this plot is part of a larger subplots, pass the figure object here, otherwise one is created
        ax: matplotlib.pyplot.axis
            if this plot is part of a larger subplots, pass the specific axis object here, otherwise one is created

        Returns
        ----------
        fig, ax : matplotlib.Figure/axis object
            objects containing the matplotlib figure and axis for further modifications if needed.
        """
        # TODO redo this to look at solution timestep zero.
        pass

    def save_plot(self, save_path, fig):
        """
        saves a matplotlib figure in `fig` to `save_path` if save_path is not None. Will fail if path is not writeable.
        Attaches model metadata to image to ensure reproducibility.

        Parameters
        ----------
        save_path: {str, None}, optional
            the save path to which to save the figure, None implies figure will not be saved.

        fig: matplotlib.pyplot.figure
            figure you are trying to save

        Returns
        ----------
        None
        """
        if save_path:
            metadata = PngInfo()
            if self.GIT_REPO.is_dirty():
                warnings.warn(
                    """\n Uncommitted Changes Warning: In order to ensure replicability of your image,
                    please commit/push your changes so that the commit
                    hash may be saved in the meta data of this image, along with config parameters. \n
                    Reproducibility is pivotal to science!"""
                )
            metadata.add_text("model", self.to_json())
            fig.savefig(save_path, pil_kwargs={"pnginfo": metadata})

    def to_json(self, file=None):
        """
        a simple method which takes self.runner_params and dumps it into `file`.
        this method effectively deals with nested numpy and jax arrays
        which are normally not JSON serializable and cause errors.
        Also able to return a string representation of the model JSON if `file` is None

        Parameters
        ----------
        `file`: TextIOWrapper | None
            a file object that can be written to, usually the result of a call like open("file.txt")

        Returns
        ----------
        None if file object is passed as parameter, str of JSON otherwise.
        """

        # define a custom encoder so that things like Enums, numpy arrays,
        # and Diffrax.Solution objects can be JSON serializable.
        # Wrap everything in a dict with the object type inside.
        class CustomEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray) or isinstance(obj, jnp.ndarray):
                    return {"type": "jax", "val": obj.tolist()}
                if isinstance(obj, EnumMeta):
                    return {
                        "type": "enum",
                        "val": {
                            str(e): idx for e, idx in zip(obj, range(len(obj)))
                        },
                    }
                # if we get a solution object, save the final-state only
                if isinstance(obj, Solution):
                    return {
                        "type": "state",
                        "val": tuple(y[-1] for y in obj.ys),
                    }
                if isinstance(obj, datetime.date):
                    return {"type": "date", "val": obj.strftime("%d-%m-%y")}
                try:
                    res = {
                        "type": "default",
                        "val": json.JSONEncoder.default(self, obj),
                    }
                except TypeError:
                    res = "error not serializable"
                return res

        # lets save the final state of the model if it has been run
        if hasattr(self, "solution"):
            self.config_file["final_state"] = self.solution
        if file:
            return json.dump(
                self.config_file, file, indent=4, cls=CustomEncoder
            )
        else:  # if given empty file, just return JSON string
            return json.dumps(self.config_file, indent=4, cls=CustomEncoder)
