"""Solve a system of ODEs and return a Solution object."""

import datetime
from collections.abc import Callable
from typing import Union

import jax
import jax.numpy as jnp
import numpyro  # type: ignore
from diffrax import (  # type: ignore
    ConstantStepSize,
    ODETerm,
    PIDController,
    SaveAt,
    Solution,
    Tsit5,
    diffeqsolve,
)
from jaxtyping import PyTree

from . import SEIC_Compartments
from .utils import date_to_sim_day

numpyro.set_host_device_count(4)
jax.config.update("jax_enable_x64", True)


class MechanisticRunner:
    """Solves ODEs using Diffrax and produces Solution objects."""

    def __init__(
        self,
        model: Callable[
            [jax.typing.ArrayLike, PyTree, dict],
            SEIC_Compartments,
        ],
    ):
        """Initialize MechanisticRunner for solving Ordinary Differential Equations.

        Parameters
        ----------
        model : Callable[[jax.typing.ArrayLike, PyTree, dict], SEIC_Compartments]
            Set of ODEs, taking time, initial state, and dictionary of
            parameters.
        """
        self.model = model

    def run(
        self,
        initial_state: SEIC_Compartments,
        args: dict,
        tf: Union[int, datetime.date] = 100,
    ) -> Solution:
        """Solve ODEs for `tf` days using `initial_state` and `args` parameters.

        Uses diffrax.Tsit5() solver.


        Parameters
        ----------
        initial_state : SEIC_Compartments
            tuple of jax arrays representing the compartments modeled by
            ODEs in their initial states at t=0.
        args : dict[str,Any]
            arguments to pass to ODEs containing necessary parameters to
            solve.
        tf : int | datetime.date, Optional
            number of days to solve ODEs for, if date is passed, runs
            up to that date, by default 100 days

        Returns
        -------
        diffrax.Solution
            Solution object, sol.ys containing compartment states for each day
            including t=0 and t=tf. For more information on whats included
            within diffrax.Solution see:
            https://docs.kidger.site/diffrax/api/solution/

        Notes
        -----
        - No partial date (or time) calculations partial days are truncated
        - if `args["CONSTANT_STEP_SIZE"] > 0` uses constant stepsizer of
        that size, else uses adaptive step sizing with
        `args["SOLVER_RELATIVE_TOLERANCE"]` and
        `args["SOLVER_ABSOLUTE_TOLERANCE"]`
        - discontinuous timepoints can not be specified with constant step sizer
        """
        term = ODETerm(
            lambda t, state, parameters: self.model(state, t, parameters)
        )
        solver = Tsit5()
        t0 = 0.0
        dt0 = 1.0
        tf_int = (
            date_to_sim_day(tf, args["INIT_DATE"])
            if isinstance(tf, datetime.date)
            else tf
        )
        assert isinstance(
            tf_int, (int, float)
        ), "tf must be of type int float or datetime.date"

        saveat = SaveAt(ts=jnp.linspace(t0, tf_int, int(tf_int) + 1))
        # jump_ts describe points in time where the model is not fully differentiable
        # this is often due to piecewise changes in parameter values like Beta
        # this is why many functions in the runner/params are required to be continuously differentiable.
        if "CONSTANT_STEP_SIZE" in args.keys() and args["CONSTANT_STEP_SIZE"]:
            dt0 = args["CONSTANT_STEP_SIZE"]
            # if user specifies they want constant step size, set it here
            stepsize_controller = ConstantStepSize()
            print("using Constant Step Size ODES with size %s" % (str(dt0)))
        else:  # otherwise use adaptive step size.
            jump_ts = (
                list(args["BETA_TIMES"])
                if "BETA_TIMES" in args.keys()
                else None
            )
            stepsize_controller = PIDController(
                rtol=args.get("SOLVER_RELATIVE_TOLERANCE", 1e-5),
                atol=args.get("SOLVER_ABSOLUTE_TOLERANCE", 1e-6),
                jump_ts=jump_ts,
            )

        solution = diffeqsolve(
            term,
            solver,
            t0,
            tf_int,
            dt0,
            initial_state,
            args=args,
            stepsize_controller=stepsize_controller,
            saveat=saveat,
            # higher for large time scales / rapid changes
            max_steps=args.get("SOLVER_MAX_STEPS", int(1e6)),
        )
        return solution
