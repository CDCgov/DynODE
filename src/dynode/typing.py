"""Module for declaring types to be used within DynODE config files."""

import datetime
import os
from datetime import date
from typing import Any, Callable, Optional, Tuple

import jax
import numpyro.distributions as dist
from jax.typing import ArrayLike

CompartmentGradiants = Tuple[jax.Array]

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


class SimulationDate(date):
    """A date object used to track simulation time.

    Meant to be used in place of a normal date when inside a Dynode CompartmentalConfig.
    """

    def __new__(cls, year, month, day):
        """Create a new SimulationDate instance."""
        return date.__new__(cls, year, month, day)

    @property
    def initialization_date(self):
        """Query the DYNODE_INITIALIZATION_DATE env variable and return it."""
        init_date = os.getenv(
            f"DYNODE_INITIALIZATION_DATE({os.getpid()})", None
        )
        if init_date is None:
            raise ValueError(
                "Reference date must be set before adding. Use set_reference_date() "
                "to set it, or with_reference_date() to create a new "
                "SimulationDate with a reference date already set."
            )
        d = datetime.datetime.strptime(init_date, "%Y-%m-%d").date()
        return d

    @property
    def sim_day(self):
        """Return the current simulation date relative to the init date."""
        difference = (self - self.initialization_date).days
        return difference

    def __add__(self, value):
        """Add a numeric value to the simulation date."""
        if isinstance(value, ArrayLike):  # type: ignore
            return self.sim_day + value
        elif isinstance(value, SimulationDate):
            return self.sim_day + value.sim_day
        return super().__add__(value)

    def __sub__(self, value):
        """Subtract a numeric value from the simulation date."""
        if isinstance(value, ArrayLike):  # type: ignore
            return self.sim_day - value
        elif isinstance(value, SimulationDate):
            return self.sim_day - value.sim_day
        return super().__sub__(value)

    def __rsub__(self, value):
        """Subtract a numeric value from the simulation date."""
        return self.__sub__(value)

    def __mul__(self, value):
        """Multiply a numeric value with the simulation date."""
        if isinstance(value, ArrayLike):  # type: ignore
            return self.sim_day * value
        elif isinstance(value, SimulationDate):
            return self.sim_day * value.sim_day
        return super().__mul__(value)

    def __rmul__(self, value):
        """Multiply a numeric value with the simulation date."""
        return self.__mul__(value)

    def __ge__(self, value):
        """Greater than or equal to comparison for simulation date."""
        if isinstance(value, ArrayLike):  # type: ignore
            return self.sim_day.__ge__(value)
        elif isinstance(value, SimulationDate):
            return self.sim_day > value.sim_day
        return super().__ge__(value)

    def __repr__(self):
        """Return a string representation of the SimulationDate."""
        return f"SimulationDate: ({self.year}-{self.month}-{self.day})({self.sim_day}) "


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
        if self.index is None:
            return self.transform(parameter_state[self.depends_on])
        else:
            return self.transform(parameter_state[self.depends_on][self.index])
