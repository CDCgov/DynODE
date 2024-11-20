"""
The following is a class which runs a series of ODE equations, and returns Solution objects for analysis or fitting.
"""

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
    Tsit5,
    diffeqsolve,
)
from jaxtyping import PyTree

from . import SEIC_Compartments
from .utils import date_to_sim_day

numpyro.set_host_device_count(4)
jax.config.update("jax_enable_x64", True)


class MechanisticRunner:
    """A class responsible for solving Ordinary Differential Equations (ODEs)
    given some initial state, parameters, and the equations themselves"""

    def __init__(
        self,
        model: Callable[
            [jax.typing.ArrayLike, PyTree, dict],
            SEIC_Compartments,
        ],
    ):
        self.model = model

    def run(
        self,
        initial_state: SEIC_Compartments,
        args: dict,
        tf: Union[int, datetime.date] = 100,
    ):
        """
        run `self.model` using `initial_state` as y@t=0 and parameters provided by the `args` dictionary.
        `self.model` will run for `tf` days if isinstance(tf, int)
        or until specified datetime if isintance(tf, datetime).

        NOTE
        --------------
        - No partial date (or time) calculations partial days are truncated down.
        - Uses date object within `args['INIT_DATE']` to calculate time between `t=0` and `t=tf`
        - if `args["CONSTANT_STEP_SIZE"] > 0` uses constant stepsizer of that size, else uses adaptive step sizing
            - discontinuous timepoints can not be specified with constant step sizer
        - implemented with `diffrax.Tsit5()` solver
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
                rtol=1e-6,
                atol=1e-8,
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
            max_steps=int(5e6),
        )
        return solution
