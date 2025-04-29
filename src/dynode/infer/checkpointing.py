import datetime

import numpyro
from diffrax import Solution

from ..config import SimulationConfig
from ..utils import date_to_sim_day


def checkpoint_compartment_sizes(
    config: SimulationConfig,
    solution: Solution,
    save_final_timesteps: bool = True,
    compartment_save_dates: list[datetime.date] = [],
):
    """Note compartment sizes at specific key dates for later debugging.

    Saves requested dates from `compartment_save_dates` if the date exists in `solution`

    Parameters
    ----------
    config : SimulationConfig
        config used to run the odes in question.

    solution : diffrax.Solution
        a diffrax Solution object returned by solving ODEs.

    """
    assert solution.ys is not None, "solution.ys returned None, odes failed."
    if save_final_timesteps:
        for compartment_name, idx in config.idx.__dict__.items():
            numpyro.deterministic(
                "final_timestep_%s" % compartment_name,
                solution.ys[idx][-1],
            )
    for date in compartment_save_dates:
        date_str = date.strftime("%Y_%m_%d")
        sim_day = date_to_sim_day(date, config.initializer.initialize_date)
        # ensure user requests a day we actually have in `solution`
        if sim_day >= 0 and sim_day < len(solution.ys[0]):
            for compartment_name, idx in config.idx.__dict__.items():
                numpyro.deterministic(
                    f"{date_str}_timestep_{compartment_name}",
                    solution.ys[idx][sim_day],
                )
