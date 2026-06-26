from __future__ import annotations

from datetime import timedelta
from typing import Any

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler


def _init_date(experiment: Any, year: int):
    key = str(year)
    for instance in experiment.spec.instances:
        if instance.key == key:
            return instance.static_context["settings"].init_date
    raise KeyError(year)


def visualize_fit(preds, new_data, year, experiment, data, lci=0.05, uci=0.95):
    ci = np.array([lci, uci])
    mean_hosp_pred = jnp.mean(preds[f"{year}_hospitalization"], axis=0)
    ci_hosp_pred = jnp.quantile(preds[f"{year}_hospitalization"], q=ci, axis=0)
    mean_subtype_pred = jnp.mean(preds[f"{year}_subtype_proportions"], axis=0)
    ci_subtype_pred = jnp.quantile(
        preds[f"{year}_subtype_proportions"], q=ci, axis=0
    )
    cumu_attack_rates = jnp.cumsum(preds[f"{year}_attack_rate_weekly"], axis=1)
    mean_cumu_attack_rate = jnp.mean(cumu_attack_rates, axis=0)
    ci_cumu_attack_rate = jnp.quantile(cumu_attack_rates, q=ci, axis=0)
    pop_immunities = jnp.array(
        [
            preds[f"{year}_immunity_h1"],
            preds[f"{year}_immunity_h3"],
            preds[f"{year}_immunity_b"],
            preds[f"{year}_immunity_all"],
        ]
    )
    pop_immunities = jnp.moveaxis(pop_immunities, 0, 2)
    mean_pop_immunities = jnp.mean(pop_immunities, axis=0)
    ci_pop_immunities = jnp.quantile(pop_immunities, q=ci, axis=0)
    init_date = _init_date(experiment, year)
    obs_data = (
        new_data[str(year)]["obs_data"]
        if str(year) in new_data
        else new_data[year]["obs_data"]
    )
    pred_hosps_dates = [
        init_date + timedelta(days=(int(d) + 1) * 7)
        for d in obs_data["obs_hosps_weeks"].tolist()
    ]
    pred_subtype_dates = [
        init_date + timedelta(days=(int(d) + 1) * 7)
        for d in obs_data["obs_subtype_weeks"].tolist()
    ]
    raw_data = data[str(year)] if str(year) in data else data[year]

    colors_age = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]
    colors_strain = ["#1b9e77", "#d95f02", "#7570b3"]
    fig, axs = plt.subplots(4, 1)
    axs[0].set_prop_cycle(cycler(color=colors_age))
    axs[0].set_ylabel("Weekly Hospitalizations")
    axs[0].set_title(str(year))
    axs[1].set_prop_cycle(cycler(color=colors_strain))
    axs[1].set_ylabel("Subtype samples")
    axs[2].set_prop_cycle(cycler(color=colors_strain))
    axs[2].set_ylabel("Cumulative attack rate")
    axs[3].set_prop_cycle(cycler(color=colors_strain + ["#555555"]))
    axs[3].set_ylabel("Population immunity")
    axs[3].set_ylim(0, 0.8)

    for low, high in zip(ci_hosp_pred[0].T, ci_hosp_pred[1].T):
        axs[0].fill_between(pred_hosps_dates, low, high, alpha=0.2)
    axs[0].plot(
        pred_hosps_dates,
        mean_hosp_pred,
        label=["0-4", "5-17", "18-49", "50-64", "65+"],
    )
    axs[0].plot(
        raw_data["dates"]["obs_hosps_dates"],
        raw_data["obs_data"]["obs_hosps_weekly"],
        linestyle=":",
    )
    axs[0].legend()

    for low, high in zip(ci_subtype_pred[0].T, ci_subtype_pred[1].T):
        axs[1].fill_between(pred_subtype_dates, low, high, alpha=0.2)
    axs[1].plot(pred_subtype_dates, mean_subtype_pred, label=["h1", "h3", "b"])
    axs[1].plot(
        raw_data["dates"]["obs_subtype_dates"],
        raw_data["obs_data"]["obs_subtype_weekly"],
        linestyle=":",
    )
    axs[1].legend()

    for low, high in zip(ci_cumu_attack_rate[0].T, ci_cumu_attack_rate[1].T):
        axs[2].fill_between(pred_subtype_dates, low, high, alpha=0.2)
    axs[2].plot(
        pred_subtype_dates, mean_cumu_attack_rate, label=["h1", "h3", "b"]
    )
    axs[2].legend()

    for low, high in zip(ci_pop_immunities[0].T, ci_pop_immunities[1].T):
        axs[3].fill_between(pred_subtype_dates, low, high, alpha=0.2)
    axs[3].plot(
        pred_subtype_dates, mean_pop_immunities, label=["h1", "h3", "b", "any"]
    )
    axs[3].legend()
    fig.set_size_inches(8, 10)

    return fig, {
        "pred_hosps_dates": [d.strftime("%y-%m-%d") for d in pred_hosps_dates],
        "mean_hosps_pred": mean_hosp_pred.tolist(),
        "obs_hosps_dates": [
            d.strftime("%y-%m-%d")
            for d in raw_data["dates"]["obs_hosps_dates"]
        ],
        "obs_hosps_weekly": raw_data["obs_data"]["obs_hosps_weekly"].tolist(),
    }
