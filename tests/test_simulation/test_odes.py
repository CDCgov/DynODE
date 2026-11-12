import chex
import jax
import jax.numpy as jnp
import pytest

import dynode.config as config
import dynode.simulation as simulation
from dynode.typing import CompartmentGradients, CompartmentState


@chex.dataclass
class TestingODEParams:
    beta: chex.ArrayDevice  # r0/infectious period
    gamma: chex.ArrayDevice  # 1/infectious period


@jax.jit
def sir_ode(
    t: float, state: CompartmentState, p: TestingODEParams
) -> CompartmentGradients:
    """A simple SIR ODE model with no time-varying components."""
    s, i, _ = state
    s_to_i = p.beta * s * i
    i_to_r = i * p.gamma
    ds = -s_to_i
    di = s_to_i - i_to_r
    dr = i_to_r
    return tuple([ds, di, dr])


@pytest.fixture
def ode_params():
    r0 = 2.0
    infectious_period = 7.0
    return TestingODEParams(
        beta=r0 / infectious_period, gamma=1 / infectious_period
    )


@pytest.fixture
def initial_state():
    return tuple([jnp.array([99.0]), jnp.array([1.0]), jnp.array([0.0])])


def test_simulation_expected_shapes(ode_params, initial_state):
    for days in [50, 100, 200, 300.0]:
        solution = simulation.simulate(
            sir_ode,
            duration_days=days,
            initial_state=initial_state,
            ode_parameters=ode_params,
            solver_parameters=config.SolverParams(),  # use default solver params
        )
        # check all compartments have appropriate shape
        assert all(
            [
                solution.ys[compartment].shape == (days + 1, 1)
                for compartment in range(len(initial_state))
            ]
        )


def test_initial_state_in_solution(ode_params, initial_state):
    solution = simulation.simulate(
        sir_ode,
        duration_days=10,
        initial_state=initial_state,
        ode_parameters=ode_params,
        solver_parameters=config.SolverParams(),  # use default solver params
    )
    initial_state_from_solution = tuple(
        [solution.ys[c][0] for c in range(len(initial_state))]
    )
    assert initial_state == initial_state_from_solution


def test_simulation_save_steps(ode_params, initial_state):
    for save_step in [1.0, 2.0, 3.0, 7.0]:
        solution = simulation.simulate(
            sir_ode,
            duration_days=100,
            initial_state=initial_state,
            ode_parameters=ode_params,
            solver_parameters=config.SolverParams(),  # use default solver params
            save_step=save_step,  # save every other day
        )
        assert all(
            [
                solution.ys[compartment].shape == (int(100 / save_step) + 1, 1)
                for compartment in range(len(initial_state))
            ]
        )


def test_subsave_indicies(ode_params, initial_state):
    # try returning all combinations of compartments
    combos = [
        [0],  # s only
        [1],  # i only
        [2],  # r only
        [0, 1],
        [0, 2],
        [1, 2],
        [0, 1, 2],  # all compartments
    ]
    for combo in combos:
        solution = simulation.simulate(
            sir_ode,
            duration_days=100,
            initial_state=initial_state,
            ode_parameters=ode_params,
            solver_parameters=config.SolverParams(),  # use default solver params
            sub_save_indices=combo,
        )
        for saved_compartment in combo:
            assert solution.ys[saved_compartment].shape == (101, 1)
        # get the compartments we didnt save, assert they are empty
        # to do this lets just take set difference from the 3 compartment indexes
        for unsaved_compartment in set([0, 1, 2]) - set(combo):
            assert solution.ys[unsaved_compartment].shape == (101, 0)  # empty
