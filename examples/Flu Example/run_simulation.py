from __future__ import annotations

from datetime import date

import jax
import jax.numpy as jnp
import numpyro

from .data import preprocess_observations
from .experiment import build_experiment
from .functions import load_vaccination_model_hill
from .specs import FluSeasonSettings

jax.config.update("jax_enable_x64", True)


def main():
    year = 2024
    init_date = date(2024, 8, 10)
    vax_data = load_vaccination_model_hill(
        "us", f"/input/data/flu/vaccination_5/{year}_{year + 1}", 5, init_date
    )
    vax_data["multipliers"] = jnp.ones(5)
    settings = {
        year: FluSeasonSettings(
            year=year,
            init_date=init_date,
            initialize_path=f"/input/data/flu/initialization/US_5b_{year}.json",
            jump_ts=tuple(float(x) for x in vax_data["t_shifts"]),
        )
    }
    experiment = build_experiment(settings)
    model = experiment.make_numpyro_model(return_outputs=True)
    data = {
        str(year): {**preprocess_observations(init_date), "vax_data": vax_data}
    }
    seeded = numpyro.handlers.seed(model, rng_seed=1234)
    out = seeded(data=data)
    print(
        out["instances"][str(year)]["observe_result"][
            "model_predictions"
        ].keys()
    )


if __name__ == "__main__":
    main()
