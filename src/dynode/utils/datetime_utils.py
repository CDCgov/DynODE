"""A module for datetime utilities in Dynode."""

import datetime

import epiweeks


def sim_day_to_date(sim_day: int, init_date: datetime.date):
    """Compute date object for given `sim_day` and `init_date`.

    Given current model's simulation day as integer and
    initialization date, returns date object representing current simulation day.

    Parameters
    ----------
    sim_day : int
        Current model simulation day where sim_day==0==init_date.

    init_date : datetime.date
        Initialization date usually found in config.INIT_DATE parameter.

    Returns
    -------
    datetime.date object representing current `sim_day`

    Examples
    --------
    >>> import datetime
    >>> init_date = datetime.date(2022, 10, 15)
    >>> sim_day_to_date(10, init_date )
    datetime.date(2022, 10, 25 )
    """
    return init_date + datetime.timedelta(days=sim_day)


def sim_day_to_epiweek(
    sim_day: int, init_date: datetime.date
) -> epiweeks.Week:
    """Calculate CDC epiweek that sim_day falls in.

    Parameters
    ----------
    sim_day : int
        Current model simulation day where sim_day==o==init_date.

    init_date : datetime.date
        Initialization date usually found in config.INIT_DATE parameter.

    Returns
    -------
    epiweeks.Week
        CDC epiweek on day sim_day

    Examples
    --------
    >>> import datetime
    >>> init_date=datetime.date(2022, 10, 15)
    >>> sim_day_to_epiweek(10, init_date )
    epiweeks.Week(year=2022, week=42)
    """
    date = sim_day_to_date(sim_day, init_date)
    epi_week = epiweeks.Week.fromdate(date)
    return epi_week


def date_to_sim_day(date: datetime.date, init_date: datetime.date):
    """Convert date object to simulation days using init_date as reference point.

    Parameters
    ----------
    date : datetime.date
        Date being converted into integer simulation days.

    init_date : datetime.date
        Initialization date usually found in config.INIT_DATE parameter.

    Returns
    -------
    int
    how many days have passed since `init _date`

    Examples
    --------
    >>> import datetime
    >>> init_date=datetime.date(2022, 10, 15)
    >>> date=datetime.date(2022, 11, 5)
    >>> date_to_sim_day(date, init_date)
    21
    """
    return (date - init_date).days


def date_to_epi_week(date: datetime.date):
    """Convert a date object to CDC epi week.

    Parameters
    ----------
    sim_day : datetime.date
        Date to be converted to a simulation day.

    Returns
    -------
    epiweeks.Week
        The epi_week that `date` falls in.
    """
    epi_week = epiweeks.Week.fromdate(date)
    return epi_week
