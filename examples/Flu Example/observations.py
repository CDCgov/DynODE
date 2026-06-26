from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from constants import (
    B_SHARE_WEIGHT,
    HOSP_LIKELIHOOD_WEIGHT,
    HOSP_NEGBIN_INF_FACTOR,
    STRAIN_NAMES,
    SUBTYPE_DIRMUL_INF_FACTOR,
    SUBTYPE_LIKELIHOOD_WEIGHT,
)

from dynode.runtime.ode_solver import solution_ys_as_state_dict
from dynode.runtime.runtime_model import RuntimeModel

B_SHARE_DIST = dist.Beta(1.0, 1.0)


def compute_observations(
    *,
    runtime: RuntimeModel,
    params: Mapping[str, Any],
    solution: Any,
    duration_days: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Map latent ODE solution to observation-scale predictions."""

    ys = solution_ys_as_state_dict(runtime=runtime, solution=solution)
    c = ys["c"]

    ihr_strain = jnp.asarray(
        [
            params["ihr_age_4_h1"],
            params["ihr_age_4_h3"],
            params["ihr_age_4_b"],
        ]
    )

    ihr_age_strain_mult = jnp.asarray(
        [
            [
                params["ihr_age_0_h1_mult"],
                params["ihr_age_0_h3_mult"],
                params["ihr_age_0_b_mult"],
            ],
            [
                params["ihr_age_1_h1_mult"],
                params["ihr_age_1_h3_mult"],
                params["ihr_age_1_b_mult"],
            ],
            [
                params["ihr_age_2_h1_mult"],
                params["ihr_age_2_h3_mult"],
                params["ihr_age_2_b_mult"],
            ],
            [
                params["ihr_age_3_h1_mult"],
                params["ihr_age_3_h3_mult"],
                params["ihr_age_3_b_mult"],
            ],
            [1.0, 1.0, 1.0],
        ]
    )

    ihr_age_strain = ihr_age_strain_mult * ihr_strain[None, :]
    vaccine_eff_matrix = params.get(
        "vaccine_eff_matrix",
        jnp.asarray([[0.0, params["ve_infection"]]] * len(STRAIN_NAMES)),
    )
    ihr_mult_vaccine = 1 - 3 * vaccine_eff_matrix[0, 1]
    ihr_vacc_mult = jnp.asarray([1.0, ihr_mult_vaccine])

    incd = jnp.diff(c, axis=0)
    incd_age_vacc_strain = jnp.sum(incd, axis=(3, 4, 5))
    ihr_age_vacc_strain = (
        ihr_age_strain[:, None, :] * ihr_vacc_mult[None, :, None]
    )

    bins = jnp.arange(len(incd) // 7, dtype=jnp.int32).repeat(7)
    remainder = len(incd) % 7

    if remainder:
        bins = jnp.append(bins, (jnp.max(bins) + 1).repeat(remainder))

    nbins = int(np.ceil(duration_days / 7))

    hosp_age_vacc_strain = incd_age_vacc_strain * ihr_age_vacc_strain
    hosp_age = jnp.sum(hosp_age_vacc_strain, axis=(2, 3))
    hosp_vacc = jnp.sum(hosp_age_vacc_strain, axis=3)
    hosp_strain = jnp.sum(hosp_age_vacc_strain, axis=(1, 2))

    sim_hosps_weekly = jnp.asarray(
        [jnp.bincount(bins, age_hosp, length=nbins) for age_hosp in hosp_age.T]
    ).T

    incd_strain = jnp.sum(incd_age_vacc_strain, axis=(1, 2))
    sim_incd_strain_weekly = jnp.asarray(
        [
            jnp.bincount(bins, strain_incd, length=nbins)
            for strain_incd in incd_strain.T
        ]
    ).T

    return (
        {
            "sim_hosps_weekly": sim_hosps_weekly,
            "sim_incd_strain_weekly": sim_incd_strain_weekly,
        },
        {
            "hosp_vacc": hosp_vacc,
            "hosp_strain": hosp_strain,
            "ys": ys,
        },
    )


def likelihood(
    *,
    sim_hosps_weekly: Any,
    sim_incd_strain_weekly: Any,
    obs_hosps_weekly: Any | None,
    obs_hosps_weeks: Any,
    obs_subtype_weekly: Any | None,
    obs_subtype_weekly_total: Any,
    obs_subtype_weeks: Any,
    year: int,
):
    """NumPyro likelihood for hospitalization and subtype observations."""

    sim_hosps_weekly_sel = sim_hosps_weekly[obs_hosps_weeks - 1] + 0.01
    negbin_concentration = sim_hosps_weekly_sel * HOSP_NEGBIN_INF_FACTOR

    sim_subtype_prop = (
        sim_incd_strain_weekly
        / (jnp.sum(sim_incd_strain_weekly, axis=1, keepdims=True) + 0.01)
        + 0.01
    )
    sim_subtype_prop_sel = sim_subtype_prop[obs_subtype_weeks]
    sim_subtype_conc_sel = (
        sim_subtype_prop_sel
        * obs_subtype_weekly_total[:, None]
        * SUBTYPE_DIRMUL_INF_FACTOR
    )

    total_strain_incd = jnp.sum(sim_incd_strain_weekly, axis=0)
    b_share = total_strain_incd[2] / (jnp.sum(total_strain_incd) + 0.01)

    with numpyro.handlers.scale(scale=HOSP_LIKELIHOOD_WEIGHT):
        numpyro.sample(
            f"{year}_hospitalization",
            dist.NegativeBinomial2(sim_hosps_weekly_sel, negbin_concentration),
            obs=obs_hosps_weekly,
        )

    with numpyro.handlers.scale(scale=SUBTYPE_LIKELIHOOD_WEIGHT):
        numpyro.sample(
            f"{year}_subtype_proportions",
            dist.DirichletMultinomial(
                concentration=sim_subtype_conc_sel,
                total_count=obs_subtype_weekly_total,
            ),
            obs=obs_subtype_weekly,
        )

    numpyro.factor(
        f"{year}_b_share_penalty",
        B_SHARE_WEIGHT * B_SHARE_DIST.log_prob(b_share),
    )

    return sim_hosps_weekly_sel


def generate_aux_output(
    *,
    runtime: RuntimeModel,
    solution: Any,
    model_predictions: Mapping[str, Any],
    extra_outcome: Mapping[str, Any],
    sim_hosps_weekly: Any,
    duration_days: int,
    year: int,
) -> None:
    """Record auxiliary deterministic outputs for posterior predictive analysis."""

    ys = extra_outcome.get("ys") or solution_ys_as_state_dict(
        runtime=runtime, solution=solution
    )

    total_pop = (
        jnp.sum(ys["s"][0])
        + jnp.sum(ys["e"][0])
        + jnp.sum(ys["i"][0])
        + jnp.sum(ys["r"][0])
    )

    pop_vacc = (
        jnp.sum(ys["s"][1:, :, 1, ...], axis=(2, 3, 4))
        + jnp.sum(ys["e"][1:, :, 1, ...], axis=(2, 3, 4, 5))
        + jnp.sum(ys["i"][1:, :, 1, ...], axis=(2, 3, 4, 5))
        + jnp.sum(ys["r"][1:, :, 1, ...], axis=(2, 3, 4, 5))
    )
    pop_unvacc = (
        jnp.sum(ys["s"][1:, :, 0, ...], axis=(2, 3, 4))
        + jnp.sum(ys["e"][1:, :, 0, ...], axis=(2, 3, 4, 5))
        + jnp.sum(ys["i"][1:, :, 0, ...], axis=(2, 3, 4, 5))
        + jnp.sum(ys["r"][1:, :, 0, ...], axis=(2, 3, 4, 5))
    )

    attack_rate_weekly = (
        model_predictions["sim_incd_strain_weekly"] / total_pop
    )
    n_weeks = int(np.ceil(duration_days / 7))
    weekend_days = (jnp.arange(n_weeks) * 7).astype(int)

    immunity_h1 = 1 - jnp.asarray(
        [
            jnp.sum(ys["s"][d][:, :, 1, :, :]) / jnp.sum(ys["s"][d])
            for d in weekend_days
        ]
    )
    immunity_h3 = 1 - jnp.asarray(
        [
            jnp.sum(ys["s"][d][:, :, :, 1, :]) / jnp.sum(ys["s"][d])
            for d in weekend_days
        ]
    )
    immunity_b = 1 - jnp.asarray(
        [
            jnp.sum(ys["s"][d][:, :, :, :, 1]) / jnp.sum(ys["s"][d])
            for d in weekend_days
        ]
    )
    immunity_all = 1 - jnp.asarray(
        [
            jnp.sum(ys["s"][d][:, :, 1, 1, 1]) / jnp.sum(ys["s"][d])
            for d in weekend_days
        ]
    )

    numpyro.deterministic(f"{year}_sim_hosps_weekly", sim_hosps_weekly)
    numpyro.deterministic(f"{year}_attack_rate_weekly", attack_rate_weekly)
    numpyro.deterministic(f"{year}_immunity_h1", immunity_h1)
    numpyro.deterministic(f"{year}_immunity_h3", immunity_h3)
    numpyro.deterministic(f"{year}_immunity_b", immunity_b)
    numpyro.deterministic(f"{year}_immunity_all", immunity_all)
    numpyro.deterministic(f"{year}_hosp_vacc", extra_outcome["hosp_vacc"])
    numpyro.deterministic(f"{year}_hosp_strain", extra_outcome["hosp_strain"])
    numpyro.deterministic(f"{year}_pop_vacc", pop_vacc)
    numpyro.deterministic(f"{year}_pop_unvacc", pop_unvacc)
