from __future__ import annotations

import json
from datetime import date

import jax
import jax.numpy as jnp
from numpyro.infer import Predictive

from .data import create_empty_observations, dict_to_json
from .experiment import build_experiment
from .functions import load_vaccination_model_hill
from .specs import FluSeasonSettings

jax.config.update("jax_enable_x64", True)

SCENARIO_MULTIPLIERS = {
    "A": jnp.array([1.0] * 5),
    "B": jnp.array([0.65] * 4 + [1.0]),
    "C": jnp.array([0.0] * 5),
}


def main():
    year = 2025
    scenario = "A"
    init_date = date(2025, 8, 9)
    vax_data = load_vaccination_model_hill(
        "us", f"/input/data/flu/vaccination_5/{year}_{year + 1}", 5, init_date
    )
    vax_data["multipliers"] = SCENARIO_MULTIPLIERS[scenario]
    settings = {
        year: FluSeasonSettings(
            year=year,
            init_date=init_date,
            initialize_path=f"/input/data/flu/initialization/US_5b_{year}.json",
            ve_infection={
                "type": "beta",
                "concentration1": 33.9,
                "concentration0": 279.1,
            },
            jump_ts=tuple(float(x) for x in vax_data["t_shifts"]),
        )
    }
    experiment = build_experiment(settings)
    posterior_samples = json.load(open("/output/flu/posterior_samples.json"))
    posterior_samples = {
        k: jnp.asarray(v) for k, v in posterior_samples.items()
    }
    data = {
        str(year): {
            **create_empty_observations(init_date),
            "vax_data": vax_data,
        }
    }
    predictive = Predictive(
        experiment.make_numpyro_model(return_outputs=False), posterior_samples
    )
    preds = predictive(jax.random.PRNGKey(1234), data=data)
    dict_to_json(preds, f"/output/flu/scenario_{scenario}_output.json")


if __name__ == "__main__":
    main()
