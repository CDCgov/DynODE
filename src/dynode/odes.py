"""Define the ODEBase class."""

from typing import Sequence, get_type_hints

import chex
import jax.numpy as jnp
from diffrax import (  # type: ignore
    AbstractStepSizeController,
    ConstantStepSize,
    ODETerm,
    PIDController,
    SaveAt,
    Solution,
    diffeqsolve,
)
from jax import Array

from dynode.typing import CompartmentGradients

from .model_configuration.params import SolverParams


@chex.dataclass
class AbstractODEParams:
    """The internal representation containing parameters passed to the ODEs.

    Because ODEs work with vectors/matricies/tensors as opposed to objects,
    this internal state flattens the list of strains into the tensors of information
    separate from the `Strain` class entirely.
    """


class ODEBase:
    """A base class defining the behavior of an ODE."""

    def __init__(self):
        """Abstract initialization method for an ODE class."""
        pass

    def solve(
        self,
        initial_state: Sequence[Array],
        solver_parameters: SolverParams,
        ode_parameters: AbstractODEParams,
        duration_days: int = 100,
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
        ode_parameters : AbstractODEParams
            ode specific parameters that dictate transmission, protection,
            strain introduction etc. Specific ODE classes will likely
            require subclasses of AbstractODEParams for their usecase.
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
        # check to make sure you are not passing the wrong AbstractODEParams
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
        t0 = 0.0
        dt0 = None  # first step size determined automatically
        assert isinstance(
            duration_days, (int, float)
        ), "tf must be of type int float or datetime.date"

        saveat = SaveAt(
            ts=jnp.linspace(t0, duration_days, int(duration_days) + 1)
        )
        stepsize_controller: AbstractStepSizeController
        if solver_parameters.constant_step_size > 0.0:
            # if user specifies they want constant step size, set it here
            stepsize_controller = ConstantStepSize()
            dt0 = solver_parameters.constant_step_size
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
            solver_parameters.solver_method,
            t0,
            duration_days,
            dt0,
            initial_state,
            args=ode_parameters,
            stepsize_controller=stepsize_controller,
            saveat=saveat,
            max_steps=solver_parameters.max_steps,
            throw=False,
        )
        return solution

    def __call__(
        self,
        compartments: Sequence[Array],
        t: float,
        p: AbstractODEParams,
    ) -> CompartmentGradients:
        """Calculate the instantanious compartment gradients for some point `t`.

        Parameters
        ----------
        compartments : Sequence[Array]
            current state of the compartments.
        t : float
            current date being calculated
        p : AbstractODEParams
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
