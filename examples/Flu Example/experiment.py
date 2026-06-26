from __future__ import annotations

from typing import Any, Mapping

import jax
from data import make_posterior_predictive_data
from numpyro.infer import Predictive
from observations import compute_observations, generate_aux_output, likelihood
from rhs import flu_rhs
from specs import FluSeasonSettings, build_flu_experiment_spec
from state import apply_flu_escape_initial_state

from dynode.runtime import DynodeExperiment, OdeSolverOptions


def build_experiment(
    settings_by_year: dict[int, FluSeasonSettings],
) -> DynodeExperiment:
    spec = build_flu_experiment_spec(settings_by_year)
    return DynodeExperiment(
        spec=spec,
        rhs_fn=flu_rhs,
        observe_fn=observe_flu_year,
        initial_state_transform=apply_flu_escape_initial_state,
        ode_solver_options=OdeSolverOptions(
            rhs_state_format="dict",
            rhs_call_style="keyword",
            validate_solution_shape=True,
        ),
        initial_state_flat=True,
    )


def observe_flu_year(
    *,
    solution: Any,
    params: Mapping[str, Any],
    data: Mapping[str, Any],
    runtime: Any,
    context: Any,
) -> dict[str, Any]:
    settings = data["settings"]
    obs_data = data.get("obs_data", {})
    model_predictions, extra_outcome = compute_observations(
        runtime=runtime,
        params=params,
        solution=solution,
        duration_days=settings.duration_days,
    )
    sim_hosps_weekly = likelihood(
        **model_predictions,
        **obs_data,
        year=settings.year,
    )
    if data.get("record_aux", False):
        generate_aux_output(
            runtime=runtime,
            solution=solution,
            model_predictions=model_predictions,
            extra_outcome=extra_outcome,
            sim_hosps_weekly=sim_hosps_weekly,
            duration_days=settings.duration_days,
            year=settings.year,
        )
    return {
        "model_predictions": model_predictions,
        "extra_outcome": extra_outcome,
        "sim_hosps_weekly": sim_hosps_weekly,
    }


def make_numpyro_model(experiment: DynodeExperiment):
    return experiment.make_numpyro_model(return_outputs=False)


def make_numpyro_model_waux(experiment: DynodeExperiment):
    def numpyro_model_waux(data: dict[str, dict[str, Any]]) -> None:
        aux_data = {
            str(key): {**value, "record_aux": True}
            for key, value in data.items()
        }
        experiment.make_numpyro_model(return_outputs=False)(aux_data)

    return numpyro_model_waux


def posterior_predictive(
    *,
    experiment: DynodeExperiment,
    posterior_samples: dict[str, Any],
    data: dict[int | str, dict[str, Any]],
    rng_key: Any | None = None,
):
    if rng_key is None:
        rng_key = jax.random.PRNGKey(1234)
    duration_days = max(
        instance.model.metadata.get("duration_days", 0)
        for instance in experiment.spec.instances
    )
    if not duration_days:
        duration_days = max(
            instance.static_context["settings"].duration_days
            for instance in experiment.spec.instances
        )
    predictive_data = make_posterior_predictive_data(data, int(duration_days))
    predictive_data = {
        str(key): {**value, "record_aux": True}
        for key, value in predictive_data.items()
    }
    predictive = Predictive(
        make_numpyro_model_waux(experiment), posterior_samples
    )
    preds = predictive(rng_key, data=predictive_data)
    return preds, predictive_data
