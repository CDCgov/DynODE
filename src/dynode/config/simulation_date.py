"""Declares the SimulationDate helper method for interpretability."""

import datetime
import os
from datetime import date


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
    for each run of the model. Use `set_dynode_init_date_flag()` to set the date.
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


def simulation_day(year: int, month: int, day: int) -> int:
    """Lookup and return the DynODE SimulationDay for this simulation.

    Parameters
    ----------
    year : int
    month : int
    day : int

    Returns
    -------
    int
        integer simulation day, can be negative if date(year, month, day) comes before
        the dynode init date.

    Raises
    ------
    ValueError
        if set_dynode_init_date_flag not called before this method to set model init date.

    Note
    ----
    use `dynode.get_dynode_init_date_flag()` to check currently set init date.
    """
    init_date = get_dynode_init_date_flag()
    if init_date is None:
        raise ValueError(
            "attempting to use SimulationDate helper method without first "
            "calling set_dynode_init_date_flag() to set env flag."
        )
    difference = (date(year, month, day) - init_date).days  # type: ignore
    return difference
