from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp
from constants import CONTACT_MATRIX_5, STRAIN_NAMES
from functions import external_i, seasonality_coswave

from dynode.runtime.layout import RuntimeModel


def flu_rhs(
    *,
    t: Any,
    y: Mapping[str, Any],
    params: Mapping[str, Any],
    runtime: RuntimeModel,
    data: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Flu SEIRS RHS using the refactored RuntimeModel state layout."""

    if data is None:
        raise ValueError(
            "flu_rhs requires data containing settings and vax_data."
        )

    settings = data["settings"]
    vax_data = dict(data["vax_data"])

    s = y["s"]
    e = y["e"]
    i = y["i"]
    r = y["r"]
    c = y["c"]

    idx = runtime.idx

    ds = jnp.zeros_like(s)
    de = jnp.zeros_like(e)
    di = jnp.zeros_like(i)
    dr = jnp.zeros_like(r)
    dc = jnp.zeros_like(c)

    n = population_by_age(
        s=s,
        e=e,
        i=i,
        r=r,
        idx=idx,
    )

    beta = (
        jnp.asarray([params[f"{strain}_r0"] for strain in STRAIN_NAMES])
        / params["strains_infectious_period"]
    )
    sigma = jnp.asarray(
        [
            1.0 / params[f"{strain}_exposed_to_infectious"]
            for strain in STRAIN_NAMES
        ]
    )
    gamma = jnp.asarray(
        [1.0 / params["strains_infectious_period"]] * len(STRAIN_NAMES)
    )

    wane_rates_h1, wane_protections_h1 = wane_arrays(runtime, "s", "waneh1")
    wane_rates_h3, wane_protections_h3 = wane_arrays(runtime, "s", "waneh3")
    wane_rates_b, wane_protections_b = wane_arrays(runtime, "s", "waneb")

    crossimmunity_matrix = crossimmunity_matrix_from_params(params)
    introduction_times = jnp.asarray(
        [params[f"{strain}_introduction_time"] for strain in STRAIN_NAMES]
    )
    introduction_scales = jnp.asarray(
        [params["introduction_scale"]] * len(STRAIN_NAMES)
    )
    introduction_percentages = jnp.asarray(
        [params["introduction_percentage"]] * len(STRAIN_NAMES)
    )
    introduction_age_masks = runtime.introduction_age_mask_matrix(
        dtype=jnp.float32
    )

    beta_modifier_params = {
        "change_points": params["beta_modifier_change_points"],
        "multipliers": params.get(
            "beta_modifier_multipliers",
            jnp.asarray([1.0, params["winter_beta_mult"], 1.0]),
        ),
    }

    if params.get("covid_change_points") is not None:
        beta_modifier_params["covid_change_points"] = params[
            "covid_change_points"
        ]
        beta_modifier_params["covid_multipliers"] = params["covid_multipliers"]

    seasonality_multiplier = seasonality_coswave(
        t,
        params["seasonality_amplitude"],
        params["seasonality_peak_day_of_year"],
        settings.init_date,
    )

    beta_multiplier = settings.beta_modifiers_func(t, **beta_modifier_params)

    external_infectious = external_i(
        t,
        introduction_times,
        introduction_scales,
        introduction_percentages,
        introduction_age_masks,
        n,
        i.shape,
    )

    infectious_contact = jnp.einsum(
        "ab,bijklm->am",
        jnp.asarray(
            settings.contact_matrix
            if settings.contact_matrix is not None
            else CONTACT_MATRIX_5
        ),
        i + external_infectious,
    )

    force_of_infection = (
        beta[None, :]
        * seasonality_multiplier
        * beta_multiplier
        * infectious_contact
    ) / n[:, None]

    foi_suscept = get_foi_suscept(
        force_of_infection=force_of_infection,
        crossimmunity_matrix=crossimmunity_matrix,
        vaccine_eff_matrix=params.get(
            "vaccine_eff_matrix",
            jnp.asarray([[0.0, params["ve_infection"]]] * len(STRAIN_NAMES)),
        ),
        wane_protections_h1=wane_protections_h1,
        wane_protections_h3=wane_protections_h3,
        wane_protections_b=wane_protections_b,
    )

    ds_to_e = jnp.moveaxis(foi_suscept * s[None, ...], 0, -1)
    ds = ds - jnp.sum(ds_to_e, axis=-1)
    dc = dc + ds_to_e

    de_to_i = e * sigma
    di_to_r = i * gamma
    dr_to_s = r * params["nu"]

    de = de + ds_to_e - de_to_i
    di = di + de_to_i - di_to_r
    dr = dr + di_to_r - dr_to_s

    ds = ds.at[:, :, 0, :, :].add(
        jnp.sum(dr_to_s[..., idx.i.strain.H1], axis=idx.i.waneh1)
    )
    ds = ds.at[:, :, :, 0, :].add(
        jnp.sum(dr_to_s[..., idx.i.strain.H3], axis=idx.i.waneh3)
    )
    ds = ds.at[:, :, :, :, 0].add(
        jnp.sum(dr_to_s[..., idx.i.strain.B], axis=idx.i.waneb)
    )

    ds = wane_immunity(s, ds, wane_rates_h1, wane_rates_h3, wane_rates_b)

    vax_data.setdefault("multipliers", jnp.ones(s.shape[0]))
    vax_rates = settings.vaccination_func(t, **vax_data)
    vax_totals = vax_rates * n
    vax_status_counts = jnp.sum(
        s, axis=(idx.s.waneh1, idx.s.waneh3, idx.s.waneb)
    )
    updated_vax_rates = vax_totals / (vax_status_counts[:, 0] + 1e-5)
    updated_vax_rates = jnp.minimum(updated_vax_rates, 0.99)
    vax_counts = s[:, 0, ...] * updated_vax_rates[:, None, None, None]

    ds = ds.at[:, 0, ...].add(-vax_counts)
    ds = ds.at[:, 1, ...].add(vax_counts)

    return {
        "s": ds,
        "e": de,
        "i": di,
        "r": dr,
        "c": dc,
    }


def population_by_age(*, s: Any, e: Any, i: Any, r: Any, idx: Any) -> Any:
    return (
        jnp.sum(s, axis=(idx.s.vacc, idx.s.waneh1, idx.s.waneh3, idx.s.waneb))
        + jnp.sum(
            e,
            axis=(
                idx.e.vacc,
                idx.e.waneh1,
                idx.e.waneh3,
                idx.e.waneb,
                idx.e.strain,
            ),
        )
        + jnp.sum(
            i,
            axis=(
                idx.i.vacc,
                idx.i.waneh1,
                idx.i.waneh3,
                idx.i.waneb,
                idx.i.strain,
            ),
        )
        + jnp.sum(
            r,
            axis=(
                idx.r.vacc,
                idx.r.waneh1,
                idx.r.waneh3,
                idx.r.waneb,
                idx.r.strain,
            ),
        )
    )


def wane_arrays(
    runtime: RuntimeModel, compartment_name: str, dimension_name: str
):
    """
    Return static waning-rate and protection vectors for one waning dimension.

    This helper intentionally uses Python ``math.isinf`` instead of
    ``jax.numpy.isinf``. The bin metadata are static model metadata, not traced
    ODE state. Using ``jnp.isinf(...)`` inside a Python ``if`` creates a JAX
    boolean tracer during Diffrax shape tracing and raises
    TracerBoolConversionError.
    """
    dimension = runtime.state_layout.get_compartment(
        compartment_name
    ).get_dimension(dimension_name)
    rates: list[float] = []
    protections: list[float] = []

    for bin_spec in dimension.bin_specs:
        waiting_time = float(getattr(bin_spec, "waiting_time"))
        rate = 0.0 if math.isinf(waiting_time) else 1.0 / waiting_time
        rates.append(rate)
        protections.append(float(getattr(bin_spec, "base_protection")))

    return jnp.asarray(rates), jnp.asarray(protections)


def crossimmunity_matrix_from_params(params: Mapping[str, Any]) -> Any:
    crossimmunity = params["crossimmunity"]
    return jnp.asarray(
        [
            [1.0, crossimmunity, crossimmunity],
            [crossimmunity, 1.0, crossimmunity],
            [crossimmunity, crossimmunity, 1.0],
        ]
    )


def get_foi_suscept(
    *,
    force_of_infection: Any,
    crossimmunity_matrix: Any,
    vaccine_eff_matrix: Any,
    wane_protections_h1: Any,
    wane_protections_h3: Any,
    wane_protections_b: Any,
):
    foi_suscept = []
    _, n_strain = force_of_infection.shape

    for strain in range(n_strain):
        foi_strain = force_of_infection[:, strain]
        crossimmunity_strain = crossimmunity_matrix[strain, :]

        waned_immunity_h1 = crossimmunity_strain[0] * wane_protections_h1
        waned_immunity_h3 = crossimmunity_strain[1] * wane_protections_h3
        waned_immunity_b = crossimmunity_strain[2] * wane_protections_b

        waned_immunity_joint = 1 - jnp.einsum(
            "i,j,k->ijk",
            1 - waned_immunity_h1,
            1 - waned_immunity_h3,
            1 - waned_immunity_b,
        )

        vax_efficacy_strain = vaccine_eff_matrix[strain, :]
        final_immunity_joint = 1 - jnp.einsum(
            "i,jkl->ijkl",
            1 - vax_efficacy_strain,
            1 - waned_immunity_joint,
        )

        foi_suscept.append(
            jnp.einsum("i,jklm->ijklm", foi_strain, 1 - final_immunity_joint)
        )

    return jnp.asarray(foi_suscept)


def _rate_vector(rate: Any, size: int) -> Any:
    rate = jnp.asarray(rate)

    if rate.ndim == 0:
        return jnp.full((size,), rate)

    if rate.shape[0] != size:
        raise ValueError(
            f"Expected rate vector of length {size}. Got shape {rate.shape}."
        )

    return rate


def wane_immunity(s: Any, ds: Any, rate_h1: Any, rate_h3: Any, rate_b: Any):
    """Move susceptible population through H1/H3/B waning dimensions."""

    rate_h1 = _rate_vector(rate_h1, s.shape[2])
    rate_h3 = _rate_vector(rate_h3, s.shape[3])
    rate_b = _rate_vector(rate_b, s.shape[4])

    s_waned_h1 = s * rate_h1[None, None, :, None, None]
    ds = ds.at[:, :, 1:, :, :].add(s_waned_h1[:, :, :-1, :, :])
    ds = ds.at[:, :, :-1, :, :].add(-s_waned_h1[:, :, :-1, :, :])

    s_waned_h3 = s * rate_h3[None, None, None, :, None]
    ds = ds.at[:, :, :, 1:, :].add(s_waned_h3[:, :, :, :-1, :])
    ds = ds.at[:, :, :, :-1, :].add(-s_waned_h3[:, :, :, :-1, :])

    s_waned_b = s * rate_b[None, None, None, None, :]
    ds = ds.at[:, :, :, :, 1:].add(s_waned_b[:, :, :, :, :-1])
    ds = ds.at[:, :, :, :, :-1].add(-s_waned_b[:, :, :, :, :-1])

    return ds
