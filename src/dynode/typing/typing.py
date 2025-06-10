"""Module for declaring types to be used within DynODE config files."""

from typing import Annotated, Callable, Tuple

import jax
from annotated_types import Ge, Le
from jaxtyping import PyTree
from pydantic import BeforeValidator
from typing_extensions import Union

CompartmentState = Tuple[jax.Array, ...]
CompartmentGradients = Tuple[jax.Array, ...]
CompartmentTimeseries = CompartmentState

UnitIntervalFloat = Annotated[float, Ge(0.0), Le(1.0)]
# an int meant to represent the day of a particular simulation
SimulationDay = int


ODE_Eqns = Callable[
    [jax.typing.ArrayLike, CompartmentState, PyTree],
    CompartmentGradients,
]

ObservedData = Union[Tuple[jax.Array, ...], jax.Array]
# an str with no spaces and no leading numbers.


def _verify_name(name: str) -> str:
    """Validate to ensure names have no spaces and dont begin with a number."""
    if name[0].isnumeric():
        raise ValueError("Name can not start with a number.")
    elif " " in name:
        raise ValueError("Name can not have spaces.")
    elif not all([char.isalnum() or char == "_" for char in name]):
        raise ValueError("Name can only contain alphanumerics or underscores.")
    return name


# a str with no spaces or leading numbers
DynodeName = Annotated[str, BeforeValidator(_verify_name)]
