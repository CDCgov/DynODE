from datetime import date, timedelta

import jax.numpy as jnp
import numpy as np
import pandas as pd


def preprocess_observations(init_date: date):
    year = init_date.year
    if year < 2020:
        obs_hosps_df = pd.read_csv(
            f"~/mounts/scenarios-mechanistic-input/data/flu/processed/fsn_usa_5_{year}.csv"
        )
    else:
        obs_hosps_df = pd.read_csv(
            f"~/mounts/scenarios-mechanistic-input/data/flu/processed/nhsn_usa_5_{year}.csv"
        )
    obs_hosps_df["weekendingdate"] = pd.to_datetime(
        obs_hosps_df["weekendingdate"].tolist()
    ).date

    obs_hosps = (
        obs_hosps_df.sort_values(["weekendingdate", "age"])
        .groupby("weekendingdate")["count"]
        .apply(np.array)
    )
    obs_hosps_dates = np.array(obs_hosps.index)
    obs_hosps_weekly = jnp.array(obs_hosps.tolist())
    obs_hosps_days = jnp.array([(d - init_date).days for d in obs_hosps_dates])
    assert all((obs_hosps_days) % 7 == 0)
    obs_hosps_weeks = (obs_hosps_days / 7 - 1).astype(int)

    obs_subtype_df = pd.read_csv(
        f"~/mounts/scenarios-mechanistic-input/data/flu/processed/subtype_prop_usa_{year}.csv"
    )
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
    obs_subtype_weekly = jnp.array(obs_subtype.tolist())
    obs_subtype_weekly = jnp.ceil(obs_subtype_weekly)
    obs_subtype_days = jnp.array(
        [(d - init_date).days for d in obs_subtype_dates]
    )
    assert all((obs_subtype_days) % 7 == 0)
    obs_subtype_weeks = (obs_subtype_days / 7 - 1).astype(int)

    data = {
        "obs_data": {
            "obs_hosps_weekly": obs_hosps_weekly,
            "obs_hosps_weeks": obs_hosps_weeks,
            "obs_subtype_weekly": obs_subtype_weekly,
            "obs_subtype_weekly_total": jnp.sum(obs_subtype_weekly, 1),
            "obs_subtype_weeks": obs_subtype_weeks,
        },
        "dates": {
            "obs_hosps_dates": obs_hosps_dates,
            "obs_subtype_dates": obs_subtype_dates,
        },
    }

    return data


def create_empty_observations(init_date: date):
    obs_hosps_weekly = jnp.array([[0, 0, 0, 0]])
    obs_hosps_weeks = np.array([0])
    obs_subtype_weekly = jnp.zeros((59, 3))
    obs_subtype_weekly_total = jnp.repeat(1000, 59)
    obs_subtype_weeks = np.arange(59)

    data = {
        "obs_data": {
            "obs_hosps_weekly": obs_hosps_weekly,
            "obs_hosps_weeks": obs_hosps_weeks,
            "obs_subtype_weekly": obs_subtype_weekly,
            "obs_subtype_weekly_total": obs_subtype_weekly_total,
            "obs_subtype_weeks": obs_subtype_weeks,
        },
        "dates": {
            "obs_hosps_dates": [
                init_date + timedelta(days=(w + 1) * 7)
                for w in obs_hosps_weeks.tolist()
            ],
            "obs_subtype_dates": [
                init_date + timedelta(days=(w + 1) * 7)
                for w in obs_subtype_weeks.tolist()
            ],
        },
    }

    return data


ves = {
    2015: 0.48,
    2016: 0.40,
    2017: 0.38,
    2018: 0.29,
    2019: 0.39,
    2022: 0.36,
    2023: 0.40,
    2024: 0.48,
    2025: 0.37,
}
