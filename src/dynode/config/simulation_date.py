"""Declares the SimulationDate class."""

import datetime
import os
from datetime import date
from functools import cached_property

from jax.typing import ArrayLike


def get_dynode_init_date_flag() -> datetime.date | None:
    """Get the dynode initialization date from the envionment variable.

    Returns
    -------
    datetime.date | None
        the date object representing the initialization date of the model in
        the current process. Or None if the environment variable is not set.

    Note
    ----
    This function uses the current process ID to ensure that the date is set
    for each run of the model. Use set_dynode_init_date_flag() to set the date.
    """
    init_date = os.getenv(f"DYNODE_INITIALIZATION_DATE({os.getpid()})", None)
    if init_date is not None:
        return datetime.datetime.strptime(init_date, "%Y-%m-%d").date()
    return None


def set_dynode_init_date_flag(init_date: datetime.date) -> None:
    """Set the dynode initialization date in the environment variable."""
    os.environ[f"DYNODE_INITIALIZATION_DATE({os.getpid()})"] = (
        init_date.strftime("%Y-%m-%d")
    )


class SimulationDate(date):
    """A date object used to track simulation time.

    Meant to be used in place of a normal date when inside a Dynode CompartmentalConfig.
    """

    def __new__(cls, year, month, day):
        """Create a new SimulationDate instance."""
        return date.__new__(cls, year, month, day)

    @cached_property
    def initialization_date(self) -> date:
        """Query the DYNODE_INITIALIZATION_DATE{os.getpid()} env variable and return it.

        Note
        ----
        This is a cached property, meaning it is executed only once per instance.

        Raises
        ------
        ValueError if the DYNODE_INITIALIZATION_DATE env variable is not set.

        Returns
        -------
        date
            The initialization date as a date object.
        """
        init_date = get_dynode_init_date_flag()
        if init_date is None:
            raise ValueError(
                "Reference date must be set before adding. Use set_reference_date() "
                "to set it, or with_reference_date() to create a new "
                "SimulationDate with a reference date already set."
            )
        return init_date

    @property
    def sim_day(self) -> int:
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
