"""Define the ODEBase class."""

from inspect import getfullargspec
from typing import get_type_hints

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

from .typing import CompartmentState, ODE_Eqns

from .model_configuration.params import SolverParams


@chex.dataclass
class AbstractODEParams:
    """The internal representation containing parameters passed to the ODEs.

    Because ODEs work with vectors/matricies/tensors as opposed to objects,
    this internal state flattens the list of strains into the tensors of information
    separate from the `Strain` class entirely.
    """


def simulate(
    ode: ODE_Eqns,
    duration_days: int,
    initial_state: CompartmentState,
    ode_parameters: AbstractODEParams,
    solver_parameters: SolverParams,
) -> Solution:
    """Solve `model` ODEs for `tf` days using `initial_state` and `args` parameters.

    Parameters
    ----------
    ode: ODEs
        a callable that takes in a numeric time, a compartment state, and the
        passed `ode_parameters` in that order and returns the gradients
        of the compartment state at that numeric time.
    initial_state : CompartmentState
        tuple of jax arrays representing the compartments modeled by
        ODEs in their initial states at t=0.
    solver_parameters : SolverParams
        solver specific parameters that dictate how the ODE solver works.
    ode_parameters : AbstractODEParams
        ode specific parameters that dictate transmission, protection,
        strain introduction etc. Specific ODE classes will likely
        require subclasses of AbstractODEParams for their usecase.
    duration_days : int, Optional
        number of days to solve ODEs for, by default 100 days

    Returns
    -------
    diffrax.Solution
        Solution object, sol.ys containing compartment states for each day
        including t=0 and t=duration_days. For more information on whats included
        within diffrax.Solution see:
        https://docs.kidger.site/diffrax/api/solution/

    Raises
    ------
    TypeError
        `initial_state` must only contain jax.Array types.
    """
    if any(
        [not isinstance(compartment, Array) for compartment in initial_state]
    ):
        raise TypeError(
            "Please pass jax.numpy.array instead of np.array to ODEs"
        )
    # check that simulate passes expected params object to `model`
    expected_ode_parameters_type = get_type_hints(ode)[
        getfullargspec(ode).args[2]
    ]
    assert type(ode_parameters) is expected_ode_parameters_type, (
        f"passed {type(ode_parameters)} ode parameters, but your ODE model "
        f"expects {expected_ode_parameters_type}"
    )
    term = ODETerm(ode)
    t0 = 0.0
    dt0 = None  # first step size determined automatically
    assert isinstance(
        duration_days, (int, float)
    ), "tf must be of type int or float"

    saveat = SaveAt(ts=jnp.linspace(t0, duration_days, int(duration_days) + 1))
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
    )
    return solution
