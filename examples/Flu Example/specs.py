from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable

import jax.numpy as jnp
import numpy as np

from dynode.distributions import (
    AffineTransformSpec,
    BetaSpec,
    HalfNormalSpec,
    TransformedDistributionSpec,
    TruncatedNormalSpec,
)
from dynode.experiment import (
    ExperimentSpec,
    ModelInstanceSpec,
)
from dynode.parameters import (
    DeterministicSpec,
    ParameterBlockSpec,
    PriorSpec,
)
from dynode.solver import (
    PIDControllerSpec,
    SaveAtSpec,
    SolverSpec,
    Tsit5Spec,
)
from dynode.structure import (
    AgeBin,
    AgeDimensionSpec,
    BinSpec,
    CompartmentInitialConditionSpec,
    CompartmentSpec,
    DimensionSpec,
    InitializerSpec,
    ModelSpec,
    WaneBin,
)
from dynode.transmission import (
    InteractionSpec,
    StrainSpec,
    TransmissionSpec,
)
from dynode.value import (
    ConstantValueSpec,
    ParamRef,
)

try:
    from .constants import (
        AGE_BIN_EDGES,
        CONTACT_MATRIX_5,
        DEFAULT_BETA_CHANGE_POINTS,
        DEFAULT_DURATION_DAYS,
        HOSPITALIZATION_VE_BY_YEAR,
        STRAIN_NAMES,
        US_POPULATION_BY_AGE_5,
        ve_hospitalization_to_ve_infection,
    )
    from .data import load_initial_s_proportions
    from .functions import beta_modifier_step, vaccination_rate_hill
except (
    ImportError
):  # Allows running files directly from this example directory.
    from constants import (
        AGE_BIN_EDGES,
        CONTACT_MATRIX_5,
        DEFAULT_BETA_CHANGE_POINTS,
        DEFAULT_DURATION_DAYS,
        HOSPITALIZATION_VE_BY_YEAR,
        STRAIN_NAMES,
        US_POPULATION_BY_AGE_5,
        ve_hospitalization_to_ve_infection,
    )
    from data import load_initial_s_proportions
    from functions import beta_modifier_step, vaccination_rate_hill


@dataclass(frozen=True, slots=True)
class EscapePriorConfig:
    escape_h1: Any = 0.0
    escape_h3: Any = 0.0
    escape_b: Any = 0.0


@dataclass(frozen=True, slots=True)
class FluSeasonSettings:
    year: int
    init_date: date
    initialize_path: str
    duration_days: int = DEFAULT_DURATION_DAYS
    population_by_age: Any = US_POPULATION_BY_AGE_5
    contact_matrix: Any = CONTACT_MATRIX_5
    ve_infection: Any | None = None
    vaccination_func: Callable[..., Any] = vaccination_rate_hill
    beta_modifiers_func: Callable[..., Any] = beta_modifier_step
    beta_change_points: Any = DEFAULT_BETA_CHANGE_POINTS
    covid_change_points: Any | None = None
    covid_multipliers: Any | None = None
    jump_ts: tuple[float, ...] = field(default_factory=tuple)
    vax_multipliers: Any | None = None
    escape: EscapePriorConfig = field(default_factory=EscapePriorConfig)

    def resolved_ve_infection(self) -> Any:
        if self.ve_infection is not None:
            return self.ve_infection
        ve_hosp = HOSPITALIZATION_VE_BY_YEAR.get(self.year)
        if ve_hosp is None:
            return 0.0
        return ve_hospitalization_to_ve_infection(ve_hosp)


def build_flu_experiment_spec(
    settings_by_year: dict[int, FluSeasonSettings],
) -> ExperimentSpec:
    shared = build_shared_parameter_block()
    instances = tuple(
        build_flu_instance_spec(
            settings=settings,
            shared_parameter_names=tuple(shared.resolved_parameter_names),
        )
        for settings in settings_by_year.values()
    )
    return ExperimentSpec(
        name="multi_year_flu",
        version="experiment-refactor-example",
        description="Seasonal flu experiment using DynODE ExperimentSpec and DynodeExperiment.",
        shared_parameters=shared,
        instances=instances,
        metadata={"example": "flu"},
    )


def build_shared_parameter_block() -> ParameterBlockSpec:
    priors: list[PriorSpec] = [
        PriorSpec(
            name="strains_infectious_period",
            distribution=TransformedDistributionSpec(
                base=BetaSpec(concentration1=4.0, concentration0=7.0),
                transforms=(
                    AffineTransformSpec(
                        loc=1.5, scale=3.5, domain="unit_interval"
                    ),
                ),
            ),
        ),
        PriorSpec(
            name="strains_r0_mu",
            distribution=TransformedDistributionSpec(
                base=BetaSpec(concentration1=2.0, concentration0=5.0),
                transforms=(
                    AffineTransformSpec(
                        loc=1.0, scale=7.0, domain="unit_interval"
                    ),
                ),
            ),
        ),
        PriorSpec(
            name="strains_r0_sigma", distribution=HalfNormalSpec(scale=1.0)
        ),
        PriorSpec(
            name="strains_0_introduction_time_mu",
            distribution=TruncatedNormalSpec(
                loc=100.0, scale=20.0, low=50.0, high=250.0
            ),
        ),
        PriorSpec(
            name="strains_1_introduction_time_mu",
            distribution=TruncatedNormalSpec(
                loc=100.0, scale=20.0, low=50.0, high=250.0
            ),
        ),
        PriorSpec(
            name="strains_2_introduction_time_mu",
            distribution=TruncatedNormalSpec(
                loc=140.0, scale=20.0, low=100.0, high=250.0
            ),
        ),
        PriorSpec(
            name="seasonality_amplitude",
            distribution=TransformedDistributionSpec(
                base=BetaSpec(concentration1=1.0, concentration0=1.0),
                transforms=(
                    AffineTransformSpec(
                        loc=0.0, scale=0.2, domain="unit_interval"
                    ),
                ),
            ),
        ),
        PriorSpec(
            name="seasonality_peak_day_of_year",
            distribution=TruncatedNormalSpec(
                loc=0.0, scale=5.0, low=-50.0, high=50.0
            ),
        ),
        PriorSpec(
            name="winter_beta_mult",
            distribution=BetaSpec(concentration1=8.0, concentration0=2.0),
        ),
    ]

    for age_idx in range(4):
        for strain_name in ("h1", "h3", "b"):
            priors.append(
                PriorSpec(
                    name=f"ihr_age_{age_idx}_{strain_name}_mult",
                    distribution=BetaSpec(
                        concentration1=5.0, concentration0=5.0
                    ),
                )
            )

    deterministic = [
        constant_det("strains_introduction_time_sigma", 25.0),
        constant_det("crossimmunity", 0.2),
        constant_det("ihr_age_4_h1", 0.008),
        constant_det("ihr_age_4_h3", 0.026),
        constant_det("ihr_age_4_b", 0.012),
    ]

    return ParameterBlockSpec(
        name="shared", priors=tuple(priors), deterministic=tuple(deterministic)
    )


def build_flu_instance_spec(
    *, settings: FluSeasonSettings, shared_parameter_names: tuple[str, ...]
) -> ModelInstanceSpec:
    local_parameters = build_year_parameter_block(
        settings=settings, shared_parameter_names=shared_parameter_names
    )
    static_parameter_context = build_static_parameter_context(settings)
    model = build_flu_model_spec(
        settings=settings,
        external_parameter_names=tuple(
            sorted(
                set(shared_parameter_names)
                | local_parameters.resolved_parameter_names
                | set(static_parameter_context)
            )
        ),
    )
    # Expose static parameter_context both as a nested object used by
    # DynodeExperiment and as top-level keys used by ExperimentSpec /
    # compile_experiment dependency validation.
    static_context = {
        **static_parameter_context,
        "settings": settings,
        "parameter_context": static_parameter_context,
    }

    return ModelInstanceSpec(
        key=str(settings.year),
        model=model,
        parameters=local_parameters,
        static_context=static_context,
        t0=0.0,
        t1=float(settings.duration_days),
        metadata={"year": str(settings.year)},
    )


def build_year_parameter_block(
    *, settings: FluSeasonSettings, shared_parameter_names: tuple[str, ...]
) -> ParameterBlockSpec:
    priors: list[PriorSpec] = []
    deterministic: list[DeterministicSpec] = []

    for idx, strain in enumerate(STRAIN_NAMES):
        priors.append(
            PriorSpec(
                name=f"{strain}_r0",
                distribution=TruncatedNormalSpec(
                    loc=ParamRef(name="strains_r0_mu"),
                    scale=ParamRef(name="strains_r0_sigma"),
                    low=1.0,
                    high=None,
                ),
            )
        )
        priors.append(
            PriorSpec(
                name=f"{strain}_introduction_time",
                distribution=TruncatedNormalSpec(
                    loc=ParamRef(name=f"strains_{idx}_introduction_time_mu"),
                    scale=ParamRef(name="strains_introduction_time_sigma"),
                    low=100.0 if strain == "B" else 50.0,
                    high=250.0,
                ),
            )
        )
        deterministic.append(
            constant_det(
                f"{strain}_exposed_to_infectious",
                0.6 if strain == "B" else 1.3,
            )
        )

    ve_value = settings.ve_infection
    if hasattr(ve_value, "type") or (
        isinstance(ve_value, dict) and "type" in ve_value
    ):
        priors.append(PriorSpec(name="ve_infection", distribution=ve_value))

    for name, value in {
        "escape_h1": settings.escape.escape_h1,
        "escape_h3": settings.escape.escape_h3,
        "escape_b": settings.escape.escape_b,
    }.items():
        if hasattr(value, "type"):
            priors.append(PriorSpec(name=name, distribution=value))
        elif isinstance(value, dict) and "type" in value:
            priors.append(PriorSpec(name=name, distribution=value))
        else:
            deterministic.append(constant_det(name, value))

    return ParameterBlockSpec(
        name=f"year_{settings.year}",
        priors=tuple(priors),
        deterministic=tuple(deterministic),
        external_dependencies=tuple(shared_parameter_names),
    )


def build_static_parameter_context(
    settings: FluSeasonSettings,
) -> dict[str, Any]:
    ve_infection = settings.resolved_ve_infection()
    context = {
        "introduction_scale": 20.0,
        "introduction_percentage": 0.004,
        "nu": 1.0 / 21.0,
        "beta_modifier_change_points": jnp.asarray(
            settings.beta_change_points
        ),
        # beta_modifier_multipliers defaults to [1, winter_beta_mult, 1] in rhs.py.
        "covid_change_points": None
        if settings.covid_change_points is None
        else jnp.asarray(settings.covid_change_points),
        "covid_multipliers": None
        if settings.covid_multipliers is None
        else jnp.asarray(settings.covid_multipliers),
    }
    if not (
        hasattr(ve_infection, "type")
        or (isinstance(ve_infection, dict) and "type" in ve_infection)
    ):
        context["ve_infection"] = ve_infection
        context["vaccine_eff_matrix"] = jnp.asarray([[0.0, ve_infection]] * 3)
    return context


def constant_det(name: str, value: Any) -> DeterministicSpec:
    return DeterministicSpec(
        name=name, expression=ConstantValueSpec(value=value)
    )


def build_flu_model_spec(
    *, settings: FluSeasonSettings, external_parameter_names: tuple[str, ...]
) -> ModelSpec:
    age_dimension = build_age_dimension()
    vacc_dimension = build_vaccination_dimension()
    waneh1_dimension = build_wane_dimension("waneh1", waiting_time=900.0)
    waneh3_dimension = build_wane_dimension("waneh3", waiting_time=900.0)
    waneb_dimension = build_wane_dimension("waneb", waiting_time=1800.0)
    strain_dimension = build_strain_dimension()
    compartments = build_compartments(
        age_dimension,
        vacc_dimension,
        waneh1_dimension,
        waneh3_dimension,
        waneb_dimension,
        strain_dimension,
    )
    initializer = build_initializer(settings=settings)
    solver = build_solver(
        duration_days=settings.duration_days, jump_ts=settings.jump_ts
    )
    transmission = build_transmission(age_dimension=age_dimension)
    return ModelSpec(
        name=f"flu_{settings.year}",
        version="experiment-refactor-example",
        description="Seasonal flu model instance for one year.",
        simulation={"initializer": initializer, "compartments": compartments},
        solver=solver,
        transmission=transmission,
        parameters=ParameterBlockSpec(name=f"model_{settings.year}"),
        external_parameter_names=external_parameter_names,
        data=None,
        metadata={
            "year": str(settings.year),
            "init_date": settings.init_date.isoformat(),
        },
    )


def build_age_dimension() -> AgeDimensionSpec:
    return AgeDimensionSpec(
        bins=tuple(AgeBin(min_value=a, max_value=b) for a, b in AGE_BIN_EDGES)
    )


def build_vaccination_dimension() -> DimensionSpec:
    return DimensionSpec(
        name="vacc", bins=(BinSpec(name="v0"), BinSpec(name="v1"))
    )


def build_wane_dimension(name: str, *, waiting_time: float) -> DimensionSpec:
    return DimensionSpec(
        name=name,
        bins=(
            WaneBin(name="w0", waiting_time=waiting_time, base_protection=1.0),
            WaneBin(name="w1", waiting_time=math.inf, base_protection=0.2),
        ),
    )


def build_strain_dimension() -> DimensionSpec:
    return DimensionSpec(
        name="strain", bins=tuple(BinSpec(name=s) for s in STRAIN_NAMES)
    )


def build_compartments(
    age_dimension,
    vacc_dimension,
    waneh1_dimension,
    waneh3_dimension,
    waneb_dimension,
    strain_dimension,
) -> tuple[CompartmentSpec, ...]:
    s_dims = (
        age_dimension,
        vacc_dimension,
        waneh1_dimension,
        waneh3_dimension,
        waneb_dimension,
    )
    eirc_dims = s_dims + (strain_dimension,)
    return (
        CompartmentSpec(name="s", dimensions=s_dims),
        CompartmentSpec(name="e", dimensions=eirc_dims),
        CompartmentSpec(name="i", dimensions=eirc_dims),
        CompartmentSpec(name="r", dimensions=eirc_dims),
        CompartmentSpec(name="c", dimensions=eirc_dims),
    )


def build_initializer(*, settings: FluSeasonSettings) -> InitializerSpec:
    s_prop_by_age = load_initial_s_proportions(settings.initialize_path)
    s = (
        jnp.asarray(s_prop_by_age)
        * jnp.asarray(settings.population_by_age)[:, None, None, None, None]
    )
    return InitializerSpec(
        compartments=(
            CompartmentInitialConditionSpec(
                compartment_name="s",
                value=ConstantValueSpec(value=np.asarray(s).tolist()),
                allow_broadcast=False,
            ),
        ),
        missing_compartment_policy="zero",
    )


def build_solver(
    *, duration_days: int, jump_ts: tuple[float, ...]
) -> SolverSpec:
    return SolverSpec(
        solver=Tsit5Spec(),
        stepsize_controller=PIDControllerSpec(rtol=1e-5, atol=1e-6),
        dt0=0.1,
        max_steps=int(1e6),
        save_at=SaveAtSpec(
            ts=tuple(float(t) for t in range(duration_days + 1)),
            t0=False,
            t1=True,
        ),
        jump_ts=tuple(float(t) for t in sorted(set(jump_ts))),
        throw=True,
    )


def build_transmission(*, age_dimension: AgeDimensionSpec) -> TransmissionSpec:
    intro_ages = tuple(age_dimension.bins[1:])
    strains = []
    for strain in STRAIN_NAMES:
        strains.append(
            StrainSpec(
                name=strain,
                r0=ParamRef(name=f"{strain}_r0"),
                infectious_period=ParamRef(name="strains_infectious_period"),
                exposed_to_infectious=ParamRef(
                    name=f"{strain}_exposed_to_infectious"
                ),
                is_introduced=True,
                introduction_time=ParamRef(name=f"{strain}_introduction_time"),
                introduction_scale=ParamRef(name="introduction_scale"),
                introduction_percentage=ParamRef(
                    name="introduction_percentage"
                ),
                introduction_ages=intro_ages,
                interactions={
                    target: InteractionSpec.deterministic("crossimmunity")
                    for target in STRAIN_NAMES
                    if target != strain
                },
            )
        )
    return TransmissionSpec(
        strains=strains,
        default_offdiag=InteractionSpec.deterministic("crossimmunity"),
        force_diag_ones=True,
    )
