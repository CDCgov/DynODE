from __future__ import annotations

import json
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import pandas as pd

DEFAULT_PROCESSED_DATA_DIR_CANDIDATES = (
    "/input/data/flu/processed",
    "~/mounts/scenarios-mechanistic-input/data/flu/processed",
)


def default_processed_data_dir() -> Path:
    """Return the default processed-data directory for the flu example.

    The environment variable FLU_PROCESSED_DATA_DIR takes precedence.
    If it is not set, use the first known directory that exists.
    If none exist, return /input/data/flu/processed so the eventual
    FileNotFoundError points at the expected container/HPC layout.
    """

    env_path = os.environ.get("FLU_PROCESSED_DATA_DIR")
    if env_path:
        return Path(env_path).expanduser()

    for candidate in DEFAULT_PROCESSED_DATA_DIR_CANDIDATES:
        path = Path(candidate).expanduser()
        if path.exists():
            return path

    return Path(DEFAULT_PROCESSED_DATA_DIR_CANDIDATES[0]).expanduser()


def load_initial_s_proportions(path: str | Path) -> jnp.ndarray:
    """Load S_prop_by_age from the initialization JSON used by the old example."""

    with open(path, "r") as f:
        payload = json.load(f)

    if "S_prop_by_age" not in payload:
        raise KeyError(
            f"Initialization file {path!s} is missing key 'S_prop_by_age'."
        )

    return jnp.asarray(payload["S_prop_by_age"])


def preprocess_observations(
    init_date: date,
    processed_data_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Load hospitalization and subtype observations for one season.

    Parameters
    ----------
    init_date:
        Season initialization date. The year determines which processed files
        are loaded.

    processed_data_dir:
        Directory containing fsn_usa_5_<year>.csv / nhsn_usa_5_<year>.csv and
        subtype_prop_usa_<year>.csv. If omitted, this uses
        FLU_PROCESSED_DATA_DIR when set, otherwise a known default path.
    """

    year = init_date.year
    processed_data_dir = (
        Path(processed_data_dir).expanduser()
        if processed_data_dir is not None
        else default_processed_data_dir()
    )

    if year < 2020:
        hosp_path = processed_data_dir / f"fsn_usa_5_{year}.csv"
    else:
        hosp_path = processed_data_dir / f"nhsn_usa_5_{year}.csv"

    subtype_path = processed_data_dir / f"subtype_prop_usa_{year}.csv"

    if not hosp_path.exists():
        raise FileNotFoundError(
            f"Hospitalization file not found: {hosp_path}. "
            "Set FLU_PROCESSED_DATA_DIR or pass processed_data_dir explicitly."
        )
    if not subtype_path.exists():
        raise FileNotFoundError(
            f"Subtype file not found: {subtype_path}. "
            "Set FLU_PROCESSED_DATA_DIR or pass processed_data_dir explicitly."
        )

    obs_hosps_df = pd.read_csv(hosp_path)
    obs_hosps_df["weekendingdate"] = pd.to_datetime(
        obs_hosps_df["weekendingdate"].tolist()
    ).date

    obs_hosps = (
        obs_hosps_df.sort_values(["weekendingdate", "age"])
        .groupby("weekendingdate")["count"]
        .apply(np.array)
    )
    obs_hosps_dates = np.array(obs_hosps.index)
    obs_hosps_weekly = jnp.asarray(obs_hosps.tolist())
    obs_hosps_days = jnp.asarray(
        [(d - init_date).days for d in obs_hosps_dates]
    )

    if not bool(jnp.all((obs_hosps_days % 7) == 0)):
        raise ValueError(
            "Hospitalization observation dates must be aligned to weekly model bins."
        )

    obs_hosps_weeks = (obs_hosps_days / 7 - 1).astype(int)

    obs_subtype_df = pd.read_csv(subtype_path)
    obs_subtype_df["date"] = pd.to_datetime(
        obs_subtype_df["date"].tolist()
    ).date

    obs_subtype = (
        obs_subtype_df.sort_values(["date", "subtype2", "subtype"])
        .groupby("date")["count"]
        .apply(np.array)
    )
    obs_subtype = obs_subtype[obs_subtype.apply(np.sum) > 0]
    obs_subtype_dates = np.array(obs_subtype.index)
    obs_subtype_weekly = jnp.ceil(jnp.asarray(obs_subtype.tolist()))
    obs_subtype_days = jnp.asarray(
        [(d - init_date).days for d in obs_subtype_dates]
    )

    if not bool(jnp.all((obs_subtype_days % 7) == 0)):
        raise ValueError(
            "Subtype observation dates must be aligned to weekly model bins."
        )

    obs_subtype_weeks = (obs_subtype_days / 7 - 1).astype(int)

    return {
        "obs_data": {
            "obs_hosps_weekly": obs_hosps_weekly,
            "obs_hosps_weeks": obs_hosps_weeks,
            "obs_subtype_weekly": obs_subtype_weekly,
            "obs_subtype_weekly_total": jnp.sum(obs_subtype_weekly, axis=1),
            "obs_subtype_weeks": obs_subtype_weeks,
        },
        "dates": {
            "obs_hosps_dates": obs_hosps_dates,
            "obs_subtype_dates": obs_subtype_dates,
        },
    }


def create_empty_observations(
    init_date: date, n_weeks: int = 59
) -> dict[str, Any]:
    """Create placeholder observations for projection runs."""

    obs_hosps_weekly = None
    obs_hosps_weeks = np.arange(n_weeks) + 1
    obs_subtype_weekly = None
    obs_subtype_weekly_total = jnp.repeat(1000, n_weeks)
    obs_subtype_weeks = np.arange(n_weeks)

    return {
        "obs_data": {
            "obs_hosps_weekly": obs_hosps_weekly,
            "obs_hosps_weeks": obs_hosps_weeks,
            "obs_subtype_weekly": obs_subtype_weekly,
            "obs_subtype_weekly_total": obs_subtype_weekly_total,
            "obs_subtype_weeks": obs_subtype_weeks,
        },
        "dates": {
            "obs_hosps_dates": [
                init_date + timedelta(days=int((week + 1) * 7))
                for week in obs_hosps_weeks.tolist()
            ],
            "obs_subtype_dates": [
                init_date + timedelta(days=int((week + 1) * 7))
                for week in obs_subtype_weeks.tolist()
            ],
        },
    }


def make_posterior_predictive_data(
    data: dict[int, dict[str, Any]],
    duration_days: int,
) -> dict[int, dict[str, Any]]:
    """Return a copy of the data dict with observations removed for prediction."""

    from copy import deepcopy

    new_data = deepcopy(data)
    n_weeks = int(np.ceil(duration_days / 7))

    for year, year_data in new_data.items():
        year_data["obs_data"]["obs_hosps_weekly"] = None
        year_data["obs_data"]["obs_subtype_weekly"] = None
        year_data["obs_data"]["obs_hosps_weeks"] = jnp.arange(n_weeks) + 1
        year_data["obs_data"]["obs_subtype_weeks"] = jnp.arange(n_weeks)

        original_total = data[year]["obs_data"]["obs_subtype_weekly_total"]
        original_weeks = data[year]["obs_data"]["obs_subtype_weeks"]
        new_total = np.zeros(n_weeks)
        new_total[np.asarray(original_weeks)] = np.asarray(original_total)
        year_data["obs_data"]["obs_subtype_weekly_total"] = jnp.asarray(
            new_total
        )

    return new_data


def dict_to_json(
    payload: dict[str, Any], save_path: str | Path, mock_chain: bool = False
) -> None:
    """JSON writer for NumPy/JAX arrays."""

    import jax

    listified: dict[str, Any] = {}

    for key, value in payload.items():
        if value is None:
            listified[key] = None
        elif isinstance(value, jax.Array):
            listified[key] = [value.tolist()] if mock_chain else value.tolist()
        elif isinstance(value, np.ndarray):
            listified[key] = [value.tolist()] if mock_chain else value.tolist()
        else:
            listified[key] = value

    with open(save_path, "w") as f:
        json.dump(listified, f, indent=None)
