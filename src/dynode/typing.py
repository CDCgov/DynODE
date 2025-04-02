"""Module for declaring types to be used within DynODE config files."""

from typing import Any, Callable, Optional, Tuple

import jax
import numpyro.distributions as dist
from annotated_types import Ge, Le
from jaxtyping import PyTree
from typing_extensions import Annotated

CompartmentState = Tuple[jax.Array, ...]
CompartmentGradients = Tuple[jax.Array, ...]

SEIC_Compartments = Tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]
# a timeseries is a tuple of compartment sizes where the leading dimension is time
# so SEIC_Timeseries has shape (tf, SEIC_Compartments.shape) for some number of timesteps tf
SEIC_Timeseries = Tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]

UnitIntervalFloat = Annotated[float, Ge(0.0), Le(1.0)]


ODE_Eqns = Callable[
    [jax.typing.ArrayLike, CompartmentState, PyTree],
    CompartmentGradients,
]


class SamplePlaceholderError(Exception):
    """A special error raised if you attempt to randomly sample a placeholder variable."""

    pass


class PlaceholderSample(dist.Distribution):
    """A parameter that draws its values from an external set of samples."""

    def __init__(self):
        """Create a PlaceholderSample distribution."""
        super().__init__()

    def sample(self, _, sample_shape=()):
        """Retrieve sample from an external set of samples.

        Raises
        ------
        SamplePosteriorError
            if sample is called outside of an in-place substitute context like
            numpyro.handlers.substitute() or numpyro.infer.Predictive.
        """
        raise SamplePlaceholderError(
            "Attempted to sample a PosteriorSample parameter outside of a "
            "Predictive() context. This likely means you did not provide "
            "posterior samples to the context via numpyro.infer.Predictive() or "
            "numpyro.handlers.substitute()."
        )


class DeterministicParameter:
    """A parameter whose value depends on a different parameter's value."""

    def __init__(
        self,
        depends_on: str,
        index: Optional[int | tuple | slice] = None,
        transform: Callable[[Any], Any] = lambda x: x,
    ):
        """Specify a linkage between this DeterministicParameter and another value.

        Parameters
        ----------
        depends_on : str
            str identifier of the parameter to which this instance is linked.
        index : Optional[int  |  tuple  |  slice], optional
            optional index in case `depends_on` is a list you wish to index,
            by default None, grabs entire list if
            `isinstance(parameter_state[depends_on], list))`.
        """
        self.depends_on = depends_on
        self.index = index
        self.transform = transform

    def resolve(self, parameter_state: dict[str, Any]) -> Any:
        """Retrieve value from `self.depends_on` from `parameter_state`.

        Marking it as deterministic within numpyro.

        Parameters
        ----------
        parameter_state : dict[str, Any]
            current parameters, must include `self.depends_on` in keys.

        Returns
        -------
        Any
            value at parameter_state[self.depends_on][self.index]

        Raises
        ------
        IndexError
            if parameter_state[self.depends_on][self.index] does not exist or attempt to
            index with tuple on type list.

        TypeError
            if parameter_state[self.depends_on] is of type list, but `self.index` is
            a tuple, you cant index a list with a tuple, only a slice.
        """
        try:
            if self.index is None:
                return self.transform(parameter_state[self.depends_on])
            else:
                return self.transform(
                    parameter_state[self.depends_on][self.index]
                )
        except Exception as e:
            if self.index is None:
                msg = (
                    f"Was unable to find {self.depends_on} within the following "
                    f"scope, make sure DeterministicParameter dependencies are "
                    f"at the top level of the configuration object. Scope: {parameter_state}"
                )
            else:
                msg = (
                    f"Was unable to find {self.depends_on}[{self.index}] within the following "
                    f"scope, make sure DeterministicParameter dependency indexes are "
                    f"correct or you are querying a list/dict-like object. "
                    f"Scope: {parameter_state}"
                )
            raise Exception(msg) from e
