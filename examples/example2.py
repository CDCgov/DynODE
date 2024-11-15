#!/usr/bin/env python3

import argparse
import functools
import os
import typing
from typing import Annotated, NamedTuple
import chex
import json
import dataclasses
from functools import wraps
from collections.abc import Callable
from jax.typing import ArrayLike
import jax
import jax.numpy as jnp
import diffrax
import numpyro

from dynode2 import validation
from dynode2.validation import Assert, Nonnegative, Positive, GreaterThan, GreaterThanOrEqualTo

### Library stuff, to be part of imaginary dynode v0.2

class CustomJSONEncoder(json.JSONEncoder):
        def default(self, o):
            if dataclasses.is_dataclass(o):
                return dataclasses.asdict(o)
            return super().default(o)

def make_probability_model[P](p: P, ddt: Callable[[P], ArrayLike]):
    # TODO: build a probability model suitable for passing to numpyro.
    pass


def simulate_ode_diffrax(ddt, p, y0, t_start = 0, t_end = 1, dt_out = 1) -> diffrax.Solution:
    """Lightweight wrapper for diffrax.diffeqsolve."""
    term = diffrax.ODETerm(ddt)
    solver = diffrax.Tsit5()

    solution = diffrax.diffeqsolve(
        (term,), solver, t0=float(t_start), t1=float(t_end), dt0=float(dt_out), y0=y0,
        # args = (p,)
    )

    return solution


### Model-specific stuff, e.g., part of a Scenarios model repo

# TODO: do type hints need to actually be jaxy, or does it not matter since they're just hints?

@chex.dataclass(frozen = True)
class Parameters:
    """
    Parameters for the model.
    
    At present, just one immutable dataclass for the probability model and the ODE model,
    representing the actual data structure to be passed to the model code.
    
    This can be loaded directly from a JSON file for simulation.
    For inference, some parameters will be random variables.

    Validation is done via typing.Annotated, which annotates a type with arbitrary metadata.
    A validator provided by the library can check validation constraints at runtime.
    """

    t_start: Annotated[int, Nonnegative] = 0
    """
    Start time for the model simulation, in model time units.

    Typically this value will be 0, but it can be nonzero if you want to be able to use a common
    model time across different runs happening at different real times, e.g.,
    the training vs. prediction timeframes.
    """

    t_end: Annotated[int, Assert("t_end > t_start")] = 1
    """
    End time for the model simulation, in model time units.
    """

    dt_output: Annotated[int, Positive] = 1
    """
    Output spacing for the model simulation, in model time units.
    """

    n_ageclasses: Annotated[int, Positive] = 1

    n_people: Annotated[int, Positive] = 1
    """
    The number of people in the population.
    """

    alpha: Annotated[float, Positive] = 0.0

    def updated(self, **kwargs):
        updated_dict = dict(self.__dict__)
        updated_dict.update(kwargs)
        Parameters(**updated_dict)
        

class ODEState(NamedTuple):
    S: ArrayLike
    """(Partially) susceptible."""

    E: ArrayLike
    """Infected, not infectious."""

    I: ArrayLike
    """Infected, infectious."""

    SE_C: ArrayLike
    """Cumulative incidence: integral over time of flow S->E.
    
    TODO: or should it be E->I?
    """

def ddt(t, y, p: Parameters) -> ODEState:
    # TODO: determine if jax jit will auto-memoize pure function return values
    # (if not, need to cache dSE() etc.)

    # Real model not this simple; dimensions get expanded/collapsed between compartments
    dS = dIS(p.alpha) - dSE()
    dE = dSE() - dEI()
    dI = dEI() - dIS(p.alpha)
    dSE_C = dSE()

    return ODEState(S = dS, E = dE, I = dI, SE_C = dSE_C)

def dIS(alpha) -> ArrayLike:
    return jnp.full((1, 1), alpha)

def dSE() -> ArrayLike:
    return jnp.zeros((1, 1))

def dEI() -> ArrayLike:
    return jnp.zeros((1, 1))

def dSE() -> ArrayLike:
    return jnp.zeros((1, 1))

def dSE() -> ArrayLike:
    return jnp.zeros((1, 1))

def example2_simulate():
    p = Parameters(
        t_start = 0,
        t_end = 100,
        alpha = 0.1,
        dt_output = 1,
        n_ageclasses = 1,
        n_people = 1
    )
    s = simulate_ode_diffrax(ddt, p, y0 = ([[0.0]], [[0.0]], [[0.0]], [[0.0]]), t_start = p.t_start, t_end = p.t_end, dt_out = 1)
    print(s)

def example2_infer():
    p = Parameters(
        t_start = 0,
        t_end = 100,
        alpha = numpyro.sample("alpha", numpyro.distributions.HalfNormal()),
        dt_output = 1,
        n_ageclasses = 1,
        n_people = 1
    )

def main():
    example2_simulate()
    example2_simulate()

if __name__ == '__main__':
    main()
