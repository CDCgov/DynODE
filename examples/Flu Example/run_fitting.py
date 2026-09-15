from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import arviz
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from data import dict_to_json, preprocess_observations
from experiment import build_experiment, posterior_predictive
from functions import beta_modifier_step_covid, load_vaccination_model_hill
from jax.example_libraries.optimizers import exponential_decay
from matplotlib.backends.backend_pdf import PdfPages
from numpyro.optim import ClippedAdam
from specs import EscapePriorConfig, FluSeasonSettings
from visualize import visualize_fit

from dynode.infer import (
    SVIProcess,  # keep your existing inference wrapper if present
)

jax.config.update("jax_enable_x64", True)

DURATION_DAYS = 364 + 49
PROCESSED_DATA_DIR = os.environ.get("FLU_PROCESSED_DATA_DIR")
YEARS = [2015, 2016, 2017, 2018, 2019, 2022, 2023, 2024, 2025]
INIT_DATES = {
    2015: date(2015, 8, 15),
    2016: date(2016, 8, 13),
    2017: date(2017, 8, 12),
    2018: date(2018, 8, 11),
    2019: date(2019, 8, 10),
    2022: date(2022, 8, 13),
    2023: date(2023, 8, 12),
    2024: date(2024, 8, 10),
    2025: date(2025, 8, 9),
}


def build_settings_and_data():
    settings_by_year = {}
    data = {}
    vax_dir = "/input/data/flu/vaccination_5"
    for yr in YEARS:
        dt = INIT_DATES[yr]
        vax_path = os.path.join(vax_dir, f"{yr}_{yr + 1}")
        vax_data = load_vaccination_model_hill(
            "us", vax_path, num_age_groups=5, init_date=dt
        )
        vax_data["multipliers"] = jnp.array([1.0] * 5)
        beta_change_points = jnp.array([0.0, 141.0, 155.0])
        jump_ts = np.unique(
            np.append(vax_data["t_shifts"], beta_change_points)
        ).tolist()
        beta_func = None
        covid_change_points = None
        covid_multipliers = None
        if yr == 2022:
            beta_func = beta_modifier_step_covid
            covid_change_points = jnp.array([0, 106, 168])
            covid_multipliers = jnp.array([1.0, 0.85, 1.0])
            jump_ts = np.unique(
                np.append(jump_ts, covid_change_points)
            ).tolist()
        elif yr == 2023:
            beta_func = beta_modifier_step_covid
            covid_change_points = jnp.array([0, 134, 161])
            covid_multipliers = jnp.array([1.0, 0.85, 1.0])
            jump_ts = np.unique(
                np.append(jump_ts, covid_change_points)
            ).tolist()
        escape = EscapePriorConfig()
        if yr == 2025:
            escape = EscapePriorConfig(
                escape_h3={
                    "type": "transformed",
                    "base": {
                        "type": "beta",
                        "concentration1": 1.0,
                        "concentration0": 1.0,
                    },
                    "transforms": (
                        {"type": "affine", "loc": 0.0, "scale": 0.4},
                    ),
                }
            )
        settings_by_year[yr] = FluSeasonSettings(
            year=yr,
            init_date=dt,
            initialize_path=f"/input/data/flu/initialization/US_5b_{yr}.json",
            duration_days=DURATION_DAYS,
            ve_infection=None,
            beta_modifiers_func=beta_func
            if beta_func is not None
            else FluSeasonSettings.__dataclass_fields__[
                "beta_modifiers_func"
            ].default,
            beta_change_points=beta_change_points,
            covid_change_points=covid_change_points,
            covid_multipliers=covid_multipliers,
            jump_ts=tuple(float(x) for x in jump_ts),
            escape=escape,
        )
        data[str(yr)] = {
            **preprocess_observations(dt, PROCESSED_DATA_DIR),
            "vax_data": vax_data,
        }
    return settings_by_year, data


def main():
    settings_by_year, data = build_settings_and_data()
    experiment = build_experiment(settings_by_year)
    numpyro_model = experiment.make_numpyro_model(return_outputs=False)

    learning_rate_schedule = exponential_decay(
        step_size=0.0001,
        decay_steps=1000,
        decay_rate=0.7,
    )

    inferer_svi = SVIProcess(
        numpyro_model=numpyro_model,
        num_iterations=3000,
        num_samples=1000,
        optimizer=ClippedAdam(step_size=learning_rate_schedule, clip_norm=5.0),
        guide_kwargs={
            "init_scale": 0.03,
        },
    )
    inferer_svi.inference_prngkey = jax.random.PRNGKey(88119)
    inferer_svi.infer(data=data)
    samples = inferer_svi.get_samples()
    sample_summary = arviz.summary(samples)
    print(sample_summary)

    preds, new_data = posterior_predictive(
        experiment=experiment, posterior_samples=samples, data=data
    )

    job_id = "flu_experiment_refactor"
    outdir = Path(f"/output/flu/{job_id}")
    outdir.mkdir(parents=True, exist_ok=True)
    pdf_pages = PdfPages(outdir / f"fit_{job_id}.pdf")
    fit_obs_all = {}
    for year in YEARS:
        fig, fit_obs = visualize_fit(preds, new_data, year, experiment, data)
        pdf_pages.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        fit_obs_all[year] = fit_obs
    pdf_pages.close()
    sample_summary.to_csv(outdir / f"fit_{job_id}.csv")
    dict_to_json(samples, outdir / "posterior_samples.json")
    dict_to_json(preds, outdir / "preds.json")
    dict_to_json(fit_obs_all, outdir / "fit_obs_all.json")


if __name__ == "__main__":
    main()
