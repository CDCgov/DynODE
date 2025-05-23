"""Define the ODEBase class."""

from inspect import getfullargspec
from typing import Optional, Tuple, get_type_hints

import chex
import jax.numpy as jnp
from diffrax import (  # type: ignore
    AbstractStepSizeController,
    ClipStepSizeController,
    ConstantStepSize,
    ODETerm,
    PIDController,
    SaveAt,
    Solution,
    SubSaveAt,
    diffeqsolve,
)
from jax import Array

from .model_configuration.params import SolverParams
from .typing import CompartmentState, ODE_Eqns


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
    sub_save_indices: Optional[Tuple[int, ...]] = None,
    save_step: int = 1,
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
    sub_save_indices : Tuple[int, ...]
        tuple of initial_state indices specifying which compartments to save states for in sol.ys.
        sub_save_indices is optional and by default set to None.
    save_step: int
        value that lets you increment your time step at which a state is saved. If for example you would like to run
        your solution for duration_days = 100 but only save a state weekly you would pass save_step = 7.
        save_step is optional by default it is set to 1 which will have no effect.

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
        stepsize_controller = ClipStepSizeController(
            controller=PIDController(
                rtol=solver_parameters.ode_solver_rel_tolerance,
                atol=solver_parameters.ode_solver_abs_tolerance,
            ),
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
        saveat=build_saveat(t0, duration_days, save_step, sub_save_indices),
        max_steps=solver_parameters.max_steps,
    )
    return solution


def build_saveat(
    start: float,
    stop: int,
    step: int = 1,
    sub_save_indices: Optional[Tuple[int, ...]] = None,
) -> SaveAt:
    """Build the SaveAt object if sub_save_indices are not None then SaveAt is built using SubSaveAt.

    Parameters
    ----------
    start : float
        initial time step or t0 for the purpose of building an array of time steps
    stop : int
        the final time step for the purpose of building an array of time steps
    step: int
        value that lets you increment your time step at which a state is saved. If for example you would like to run
        your solution for duration_days = 100 but only save a state weekly you would pass step = 7.
        save_step is optional by default it is set to 1 which will have no effect.
    sub_save_indices : Tuple[int, ...]
        tuple of initial_state indices specifying which compartments to save states for in the final sol.ys.
        sub_save_indices is optional and by default set to None.

    Returns
    -------
    diffrax.SaveAt
        SaveAt object, which specifies which compartments and the time step they should be saved for the Solution object.
        For more information on what's included within diffrax.SaveAt see:
        https://docs.kidger.site/diffrax/api/saveat/
    """
    if step <= 0:
        step = 1
    save_times = jnp.linspace(start, stop, int(stop // step) + 1)
    built_saveat = SaveAt(ts=save_times)

    if sub_save_indices is not None:
        try:
            sub_save = SubSaveAt(
                ts=save_times,
                fn=lambda t, y, args: tuple(
                    y[i]
                    if i in sub_save_indices
                    else jnp.array([], dtype=y[i].dtype)
                    for i in range(len(y))
                ),
            )
            built_saveat = SaveAt(subs=sub_save)
        except IndexError as ex:
            print(
                f"An index passed to sub_save_indices was out of range for initial_state values. Exception: {ex}"
            )
    return built_saveat
