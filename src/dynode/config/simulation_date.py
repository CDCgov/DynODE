"""Declares the SimulationDate class."""

import datetime
import os
from datetime import date
from functools import cached_property
from typing import Any

from numpy import ndarray
from numpyro.distributions import Distribution
from pydantic import BaseModel


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
        # mypy complains on this line since `self` uses super().__sub__()
        difference = (self - self.initialization_date).days  # type: ignore
        return difference


def replace_simulation_dates(obj: Any):
    """Replace instances of SimulationDate with integer sim day.

    Parameters
    ----------
    obj : Any
        Object that may or may not be an instance of SimulationDate or list
        type containing SimulationDates

    Returns
    -------
    Any | int
        obj untouched unless is instance of SimulationDate or contains
        SimulationDate, in which case replaced by int sim day.

    Raises
    ------
    ValueError
        if this method is called outside of a SimulationConfig class which
        calls set_dynode_init_date_flag().
    """
    if isinstance(obj, SimulationDate):
        return obj.sim_day
    elif isinstance(obj, (list, ndarray)):
        for i in range(len(obj)):
            obj[i] = replace_simulation_dates(obj[i])
    elif isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = replace_simulation_dates(value)
    elif isinstance(obj, BaseModel):
        obj_dict = dict(obj)
        for key, value in obj_dict.items():
            setattr(obj, key, replace_simulation_dates(value))
            obj_dict[key] = replace_simulation_dates(value)
    elif issubclass(type(obj), Distribution):
        # sometimes distributions use simulation date as their mean.
        obj_dict = replace_simulation_dates(obj.__dict__)
        obj.__dict__ = obj_dict
    return obj
