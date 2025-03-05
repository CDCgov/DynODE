"""Define the ODEBase class."""

import datetime
from typing import Sequence, Union, get_type_hints

import jax.numpy as jnp
from diffrax import (  # type: ignore
    AbstractStepSizeController,
    ConstantStepSize,
    ODETerm,
    PIDController,
    SaveAt,
    Solution,
    Tsit5,
    diffeqsolve,
)
from jax import Array

from dynode.model_configuration import CompartmentalModel
from dynode.typing import CompartmentGradients
from dynode.utils import date_to_sim_day

from .params import ODEParameters, SolverParams


class ODEBase:
    """A base class defining the behavior of an ODE."""

    def __init__(self, compartmental_model: CompartmentalModel):
        """Abstract initialization method for an ODE class.

        Parameters
        ----------
        compartmental_model : CompartmentalModel
            CompartmentalModel that will be calling this ODE.
        """
        self.compartmental_model = compartmental_model
        pass

    def solve(
        self,
        initial_state: Sequence[Array],
        solver_parameters: SolverParams,
        ode_parameters: ODEParameters,
        tf: Union[int, datetime.date] = 100,
    ) -> Solution:
        """Solve ODEs for `tf` days using `initial_state` and `args` parameters.

        Uses diffrax.Tsit5() solver.


        Parameters
        ----------
        initial_state : Sequence[Array]
            tuple of jax arrays representing the compartments modeled by
            ODEs in their initial states at t=0.
        solver_parameters : SolverParams
            solver specific parameters that dictate how the ODE solver works.
        ode_parameters : ODEParameters
            ode specific parameters that dictate transmission, protection,
            strain introduction etc. Specific ODE classes will likely
            require subclasses of ODEParameters for their usecase.
        tf : int | datetime.date, Optional
            number of days to solve ODEs for, if datetime.date is passed, runs
            up to that date, by default 100 days

        Returns
        -------
        diffrax.Solution
            Solution object, sol.ys containing compartment states for each day
            including t=0 and t=tf. For more information on whats included
            within diffrax.Solution see:
            https://docs.kidger.site/diffrax/api/solution/

        Raises
        ------
        TypeError
            `initial_state` must only contain jax.Array types.
        """
        if any(
            [
                not isinstance(compartment, Array)
                for compartment in initial_state
            ]
        ):
            raise TypeError(
                "Please pass jax.numpy.array instead of np.array to ODEs"
            )
        # check to make sure you are not passing the wrong ODEParameters
        # subclass to self.__call__()
        expected_ode_parameters_type = get_type_hints(self.__call__)["p"]
        assert (
            type(ode_parameters) is expected_ode_parameters_type
        ), f"""passed {type(ode_parameters)} ode parameters, but your ODE
            expects {expected_ode_parameters_type}"""
        term = ODETerm(
            lambda t, state, ode_parameters: self(
                state,
                t,  # type: ignore[arg-type]
                ode_parameters,
            )
        )
        solver = Tsit5()
        t0 = 0.0
        dt0 = solver_parameters.constant_step_size
        tf_int = (
            date_to_sim_day(
                tf, self.compartmental_model.initializer.initialize_date
            )
            if isinstance(tf, datetime.date)
            else tf
        )
        assert isinstance(
            tf_int, (int, float)
        ), "tf must be of type int float or datetime.date"

        saveat = SaveAt(ts=jnp.linspace(t0, tf_int, int(tf_int) + 1))
        stepsize_controller: AbstractStepSizeController
        if dt0 > 0.0:
            # if user specifies they want constant step size, set it here
            stepsize_controller = ConstantStepSize()
        else:  # otherwise use adaptive step size.
            jump_ts = (
                jnp.array(solver_parameters.discontinuity_points)
                if len(solver_parameters.discontinuity_points) > 0
                else None
            )
            stepsize_controller = PIDController(
                rtol=solver_parameters.ode_solver_rel_tolerance,
                atol=solver_parameters.ode_solver_abs_tolerance,
                jump_ts=jump_ts,
            )

        solution = diffeqsolve(
            term,
            solver,
            t0,
            tf_int,
            dt0,
            initial_state,
            args=ode_parameters,
            stepsize_controller=stepsize_controller,
            saveat=saveat,
            max_steps=solver_parameters.max_steps,
        )
        return solution

    def __call__(
        self,
        compartments: Sequence[Array],
        t: float,
        p: ODEParameters,
    ) -> CompartmentGradients:
        """Calculate the instantanious compartment gradients for some point `t`.

        Parameters
        ----------
        compartments : Sequence[Array]
            current state of the compartments.
        t : float
            current date being calculated
        p : ODEParameters
            parameters needed to calculated gradients.

        Returns
        -------
        CompartmentGradients
            sequence of `jax.Array` with same shapes as passed in `compartments`
            but values representing the change in each cell at time `t`.
        """
        raise NotImplementedError(
            "Implement your instantanious derivative function within __call__()"
        )
