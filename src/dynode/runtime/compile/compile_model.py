from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from dynode.runtime.layout.runtime_model import (
    RuntimeModel,
    RuntimeParameterLayout,
    RuntimeTransmission,
    StateLayout,
)


class CompileError(ValueError):
    """
    Raised when a validated ModelSpec cannot be compiled into a RuntimeModel.
    """


@dataclass(frozen=True, slots=True)
class CompileOptions:
    """
    Options controlling static runtime compilation.

    These are intentionally conservative. The compiler should catch as many
    static errors as possible before NumPyro/JAX/Diffrax execution begins.
    """

    run_spec_validation_hooks: bool = True
    validate_runtime_layout: bool = True
    validate_parameter_dependencies: bool = True
    validate_data_dependencies: bool = True
    validate_transmission_dependencies: bool = True
    validate_initializer_dependencies: bool = True
    validate_data_spec: bool = True

    # For the current architecture, prior distribution parameters may depend on
    # earlier sampled priors, but not on deterministic parameters. Deterministic
    # parameters are resolved after priors.
    allow_prior_dependencies_on_deterministics: bool = False

    # Optional metadata merged into RuntimeModel.metadata.
    metadata: Mapping[str, str] = field(default_factory=dict)

    # Names supplied by an outer experiment context. These are accepted when
    # validating model-local references.
    external_parameter_names: frozenset[str] = field(default_factory=frozenset)


def compile_model(
    spec: Any,
    *,
    options: CompileOptions | None = None,
) -> RuntimeModel:
    """
    Compile a validated ModelSpec into a RuntimeModel.

    This function is the main bridge between the declarative Pydantic spec layer
    and the runtime execution layer.

    Parameters
    ----------
    spec:
        A ModelSpec-like object with:
        - simulation
        - parameters
        - solver
        - transmission
        - optional data

    options:
        Static compilation options.

    Returns
    -------
    RuntimeModel
        Compiled runtime model containing state layout, parameter layout, and
        transmission layout.

    Notes
    -----
    This function should not:
    - call numpyro.sample
    - resolve deterministic parameters numerically
    - build initial state y0
    - call diffrax.diffeqsolve
    """
    options = options or CompileOptions()

    _validate_required_model_shape(spec)

    if options.run_spec_validation_hooks:
        _run_spec_validation_hooks(spec, options=options)

    state_layout = _compile_state_layout(spec.simulation)

    parameter_layout = _compile_parameter_layout(
        spec.parameters,
        options=options,
    )

    transmission = _compile_transmission(
        spec.transmission,
        simulation=spec.simulation,
    )

    runtime = RuntimeModel(
        spec=spec,
        state_layout=state_layout,
        parameter_layout=parameter_layout,
        transmission=transmission,
        metadata=_compile_metadata(spec, options),
    )

    _validate_runtime_model(runtime, options=options)

    return runtime


def compile_model_from_dict(
    data: Mapping[str, Any],
    model_spec_type: type[Any],
    *,
    options: CompileOptions | None = None,
) -> RuntimeModel:
    """
    Convenience helper for loading a RuntimeModel from a dict.

    Example
    -------
    raw = yaml.safe_load(open("model.yaml"))
    runtime = compile_model_from_dict(raw, ModelSpec)
    """
    if not hasattr(model_spec_type, "model_validate"):
        raise TypeError(
            "model_spec_type must be a Pydantic v2 model class exposing "
            "model_validate(...)."
        )

    spec = model_spec_type.model_validate(data)

    return compile_model(
        spec,
        options=options,
    )


def _validate_required_model_shape(spec: Any) -> None:
    required_top_level = ("simulation", "parameters", "solver", "transmission")

    for attr_name in required_top_level:
        if not hasattr(spec, attr_name):
            raise CompileError(
                f"Model spec is missing required attribute {attr_name!r}."
            )

    if not hasattr(spec.simulation, "compartments"):
        raise CompileError(
            "Model spec simulation is missing required attribute 'compartments'."
        )

    if not hasattr(spec.simulation, "initializer"):
        raise CompileError(
            "Model spec simulation is missing required attribute 'initializer'."
        )


def _run_spec_validation_hooks(
    spec: Any,
    *,
    options: CompileOptions,
) -> None:
    """
    Run optional validation hooks exposed by specs.

    Most validation should already have happened during Pydantic construction.
    These hooks are useful when a nested spec wants to validate itself against
    the full model.
    """
    validate_cross_links = getattr(spec, "validate_cross_links", None)

    if callable(validate_cross_links):
        validate_cross_links()

    initializer = spec.simulation.initializer

    validate_initializer_against_model = getattr(
        initializer,
        "validate_against_model",
        None,
    )

    if callable(validate_initializer_against_model):
        validate_initializer_against_model(spec)
    else:
        validate_initializer_against_simulation = getattr(
            initializer,
            "validate_against_simulation",
            None,
        )

        if callable(validate_initializer_against_simulation):
            validate_initializer_against_simulation(spec.simulation)

    data = getattr(spec, "data", None)

    if data is not None and options.validate_data_spec:
        validate_data_against_model = getattr(
            data,
            "validate_against_model",
            None,
        )

        if callable(validate_data_against_model):
            validate_data_against_model(spec)
        else:
            validate_data_against_simulation = getattr(
                data,
                "validate_against_simulation",
                None,
            )

            if callable(validate_data_against_simulation):
                validate_data_against_simulation(spec.simulation)

    deterministic_execution_order = getattr(
        spec.parameters,
        "deterministic_execution_order",
        None,
    )

    if callable(deterministic_execution_order):
        deterministic_execution_order()

    interaction_matrix_spec = getattr(
        spec.transmission,
        "interaction_matrix_spec",
        None,
    )

    if callable(interaction_matrix_spec):
        interaction_matrix_spec()


def _compile_state_layout(simulation: Any) -> StateLayout:
    """
    Compile SimulationSpec into StateLayout.
    """
    try:
        return StateLayout.from_simulation(simulation)
    except Exception as exc:
        raise CompileError(
            "Failed to compile StateLayout from SimulationSpec."
        ) from exc


def _compile_parameter_layout(
    parameters: Any,
    *,
    options: CompileOptions,
) -> RuntimeParameterLayout:
    """
    Compile ParameterBlockSpec-like input into RuntimeParameterLayout.

    Priors are topologically ordered when prior distributions depend on
    previously sampled priors or on externally supplied/shared parameters.
    """
    priors = _get_sequence_attr(
        parameters,
        ("priors", "prior_specs"),
    )

    deterministic = _get_sequence_attr(
        parameters,
        ("deterministic", "deterministics", "deterministic_params"),
    )

    prior_order = _prior_execution_order(
        priors,
        deterministic=deterministic,
        options=options,
    )

    deterministic_order = _deterministic_execution_order(
        parameters,
        deterministic,
    )

    return RuntimeParameterLayout(
        prior_names=tuple(_name_of(prior) for prior in prior_order),
        deterministic_names=tuple(_name_of(spec) for spec in deterministic),
        deterministic_order=tuple(deterministic_order),
        prior_specs={_name_of(prior): prior for prior in prior_order},
        deterministic_specs={_name_of(spec): spec for spec in deterministic},
    )


def _compile_transmission(
    transmission: Any,
    *,
    simulation: Any,
) -> RuntimeTransmission:
    """
    Compile TransmissionSpec into RuntimeTransmission.
    """
    age_bins = _age_bins_from_simulation(simulation)

    try:
        return RuntimeTransmission.from_spec(
            transmission,
            age_bins=age_bins,
        )
    except Exception as exc:
        raise CompileError(
            "Failed to compile RuntimeTransmission from TransmissionSpec."
        ) from exc


def _validate_runtime_model(
    runtime: RuntimeModel,
    *,
    options: CompileOptions,
) -> None:
    errors: list[str] = []

    validators = []

    if options.validate_runtime_layout:
        validators.append(_validate_runtime_layout)

    if options.validate_parameter_dependencies:
        validators.append(_validate_parameter_dependencies)

    if options.validate_initializer_dependencies:
        validators.append(_validate_initializer_dependencies)

    if options.validate_data_dependencies:
        validators.append(_validate_data_dependencies)

    if options.validate_transmission_dependencies:
        validators.append(_validate_transmission_dependencies)

    for validator in validators:
        try:
            validator(runtime, options=options)
        except CompileError as exc:
            errors.append(str(exc))

    if errors:
        raise CompileError(
            "Runtime model compilation failed:\n"
            + "\n".join(f"- {error}" for error in errors)
        )


def _validate_runtime_layout(
    runtime: RuntimeModel,
    *,
    options: CompileOptions,
) -> None:
    del options

    simulation = runtime.simulation_spec
    state_layout = runtime.state_layout

    simulation_compartment_names = tuple(
        str(name) for name in getattr(simulation, "compartment_names", ())
    )

    if simulation_compartment_names:
        if state_layout.compartment_names != simulation_compartment_names:
            raise CompileError(
                "StateLayout compartment names do not match SimulationSpec. "
                f"StateLayout={state_layout.compartment_names}, "
                f"SimulationSpec={simulation_compartment_names}."
            )

    total_state_size = getattr(simulation, "total_state_size", None)

    if total_state_size is not None:
        if int(total_state_size) != state_layout.total_size:
            raise CompileError(
                "StateLayout total size does not match SimulationSpec. "
                f"StateLayout={state_layout.total_size}, "
                f"SimulationSpec={total_state_size}."
            )

    compartment_shapes = getattr(simulation, "compartment_shapes", None)

    if callable(compartment_shapes):
        expected_shapes = {
            str(name): tuple(shape)
            for name, shape in compartment_shapes().items()
        }

        actual_shapes = {
            compartment.name: compartment.shape
            for compartment in state_layout.compartments
        }

        if expected_shapes != actual_shapes:
            raise CompileError(
                "StateLayout compartment shapes do not match SimulationSpec. "
                f"StateLayout={actual_shapes}, SimulationSpec={expected_shapes}."
            )


def _validate_parameter_dependencies(
    runtime: RuntimeModel,
    *,
    options: CompileOptions,
) -> None:
    available = _available_parameter_names(runtime, options=options)
    errors: list[str] = []

    for prior_name, prior in runtime.parameter_layout.prior_specs.items():
        deps = _dependency_set(prior, "dependencies")
        missing = sorted(deps - available)

        if missing:
            errors.append(
                f"Prior {prior_name!r} depends on unknown parameters {missing}."
            )

    for (
        deterministic_name,
        deterministic,
    ) in runtime.parameter_layout.deterministic_specs.items():
        deps = _dependency_set(deterministic, "dependencies")
        missing = sorted(deps - available)

        if missing:
            errors.append(
                f"Deterministic parameter {deterministic_name!r} depends on "
                f"unknown parameters {missing}."
            )

    if errors:
        raise CompileError("; ".join(errors))


def _validate_initializer_dependencies(
    runtime: RuntimeModel,
    *,
    options: CompileOptions,
) -> None:
    initializer = runtime.initializer_spec
    available = _available_parameter_names(runtime, options=options)

    deps = _dependency_set(initializer, "dependencies")
    missing = sorted(deps - available)

    if missing:
        raise CompileError(
            "Initializer refers to unknown parameters or deterministic values: "
            f"{missing}."
        )


def _validate_data_dependencies(
    runtime: RuntimeModel,
    *,
    options: CompileOptions,
) -> None:
    del options

    data = runtime.data_spec
    available_data = _available_data_names(data)

    errors: list[str] = []

    initializer_data_deps = _dependency_set(
        runtime.initializer_spec,
        "data_dependencies",
    )

    if initializer_data_deps:
        if data is None:
            errors.append(
                "Initializer has data dependencies but ModelSpec.data is None. "
                f"Dependencies: {sorted(initializer_data_deps)}."
            )
        else:
            missing = sorted(initializer_data_deps - available_data)

            if missing:
                errors.append(
                    "Initializer refers to unknown data series: "
                    f"{missing}. Available data series are: {sorted(available_data)}."
                )

    for prior_name, prior in runtime.parameter_layout.prior_specs.items():
        data_deps = _dependency_set(prior, "data_dependencies")

        if data_deps:
            errors.append(
                f"Prior {prior_name!r} has data dependencies, which are not "
                f"supported in the current parameter sampling order: "
                f"{sorted(data_deps)}."
            )

    for (
        deterministic_name,
        deterministic,
    ) in runtime.parameter_layout.deterministic_specs.items():
        data_deps = _dependency_set(deterministic, "data_dependencies")

        if data_deps:
            if data is None:
                errors.append(
                    f"Deterministic parameter {deterministic_name!r} has data "
                    f"dependencies but ModelSpec.data is None: {sorted(data_deps)}."
                )
            else:
                missing = sorted(data_deps - available_data)

                if missing:
                    errors.append(
                        f"Deterministic parameter {deterministic_name!r} refers "
                        f"to unknown data series {missing}. Available data series "
                        f"are: {sorted(available_data)}."
                    )

    if errors:
        raise CompileError("; ".join(errors))


def _validate_transmission_dependencies(
    runtime: RuntimeModel,
    *,
    options: CompileOptions,
) -> None:
    available = _available_parameter_names(runtime, options=options)
    errors: list[str] = []

    for strain_name, strain in runtime.transmission.strain_specs.items():
        deps = _dependency_set(strain, "dependencies")
        missing = sorted(deps - available)

        if missing:
            errors.append(
                f"Strain {strain_name!r} depends on unknown parameters "
                f"{missing}."
            )

        data_deps = _dependency_set(strain, "data_dependencies")

        if data_deps:
            errors.append(
                f"Strain {strain_name!r} has data dependencies, which are not "
                f"allowed in transmission specs: {sorted(data_deps)}."
            )

    for source_name, row in zip(
        runtime.transmission.strain_names,
        runtime.transmission.interaction_matrix_spec,
    ):
        for target_name, interaction in zip(
            runtime.transmission.strain_names,
            row,
        ):
            deps = _dependency_set(interaction, "dependencies")
            missing = sorted(deps - available)

            if missing:
                errors.append(
                    f"Interaction {source_name!r}->{target_name!r} depends "
                    f"on unknown parameters {missing}."
                )

            data_deps = _dependency_set(interaction, "data_dependencies")

            if data_deps:
                errors.append(
                    f"Interaction {source_name!r}->{target_name!r} has data "
                    f"dependencies, which are not allowed in transmission specs: "
                    f"{sorted(data_deps)}."
                )

    if errors:
        raise CompileError("; ".join(errors))


def _prior_execution_order(
    priors: Sequence[Any],
    *,
    deterministic: Sequence[Any],
    options: CompileOptions,
) -> tuple[Any, ...]:
    """
    Determine prior sampling order.

    The default case is simple: priors have no dependencies, so the original
    order is preserved.

    If a prior distribution depends on another sampled prior, this function
    orders priors topologically. Dependencies on names supplied by
    CompileOptions.external_parameter_names are treated as already resolved.
    """
    if not priors:
        return tuple()

    prior_names = tuple(_name_of(prior) for prior in priors)
    deterministic_names = tuple(_name_of(spec) for spec in deterministic)

    duplicates = _duplicates(prior_names)

    if duplicates:
        raise CompileError(f"Duplicate prior names: {duplicates}.")

    prior_name_set = set(prior_names)
    deterministic_name_set = set(deterministic_names)
    external_name_set = {
        str(name) for name in options.external_parameter_names
    }

    prior_by_name = {_name_of(prior): prior for prior in priors}

    for prior_name, prior in prior_by_name.items():
        deps = _dependency_set(prior, "dependencies")

        if prior_name in deps:
            raise CompileError(
                f"Prior {prior_name!r} cannot depend on itself."
            )

        deterministic_deps = deps & deterministic_name_set

        if (
            deterministic_deps
            and not options.allow_prior_dependencies_on_deterministics
        ):
            raise CompileError(
                f"Prior {prior_name!r} depends on deterministic parameters "
                f"{sorted(deterministic_deps)}, but deterministic parameters "
                "are resolved after priors in the current runtime design."
            )

        unknown = (
            deps - prior_name_set - deterministic_name_set - external_name_set
        )

        if unknown:
            raise CompileError(
                f"Prior {prior_name!r} depends on unknown parameters "
                f"{sorted(unknown)}."
            )

    remaining = dict(prior_by_name)

    resolved: set[str] = set(external_name_set)

    if options.allow_prior_dependencies_on_deterministics:
        resolved |= deterministic_name_set

    ordered: list[Any] = []

    while remaining:
        ready_names = [
            name
            for name in prior_names
            if name in remaining
            and _dependency_set(remaining[name], "dependencies") <= resolved
        ]

        if not ready_names:
            unresolved = {
                name: sorted(_dependency_set(prior, "dependencies") - resolved)
                for name, prior in remaining.items()
            }

            raise CompileError(
                "Could not determine prior sampling order. This usually means "
                "there is a cyclic prior dependency or a prior depends on a "
                "deterministic value that is resolved after priors. "
                f"Remaining dependencies: {unresolved}."
            )

        for name in ready_names:
            prior = remaining.pop(name)
            ordered.append(prior)
            resolved.add(name)

    return tuple(ordered)


def _deterministic_execution_order(
    parameters: Any,
    deterministic: Sequence[Any],
) -> tuple[Any, ...]:
    deterministic_execution_order = getattr(
        parameters,
        "deterministic_execution_order",
        None,
    )

    if callable(deterministic_execution_order):
        try:
            return tuple(deterministic_execution_order())
        except Exception as exc:
            raise CompileError(
                "Failed to determine deterministic parameter execution order."
            ) from exc

    return tuple(deterministic)


def _age_bins_from_simulation(simulation: Any) -> tuple[Any, ...]:
    get_age_bins = getattr(simulation, "get_age_bins", None)

    if callable(get_age_bins):
        return tuple(get_age_bins())

    age_bins: list[Any] = []

    flatten_unique_dims = getattr(simulation, "flatten_unique_dims", None)

    if callable(flatten_unique_dims):
        for dimension in flatten_unique_dims():
            bins = tuple(getattr(dimension, "bins", ()))

            if bins and all(
                bin_.__class__.__name__ == "AgeBin" for bin_ in bins
            ):
                age_bins.extend(bins)
                break

    return tuple(age_bins)


def _available_parameter_names(
    runtime: RuntimeModel,
    *,
    options: CompileOptions | None = None,
) -> set[str]:
    """
    Names available for resolving parameter references during compilation.

    Includes:
    - model-local prior names
    - model-local deterministic parameter names
    - optional external/shared names supplied by an outer experiment context
    """
    parameter_layout = runtime.parameter_layout
    names: set[str] = set()

    for attr_name in (
        "prior_names",
        "deterministic_names",
        "resolved_names",
        "resolved_parameter_names",
    ):
        value = getattr(parameter_layout, attr_name, None)

        if value is None:
            continue

        if callable(value):
            value = value()

        names |= {str(name) for name in value}

    if options is not None:
        names |= {str(name) for name in options.external_parameter_names}

    return names


def _available_data_names(data: Any | None) -> set[str]:
    if data is None:
        return set()

    if isinstance(data, Mapping):
        names = set(data)

        observations = data.get("observations")
        if isinstance(observations, Mapping):
            names |= set(observations)

        covariates = data.get("covariates")
        if isinstance(covariates, Mapping):
            names |= set(covariates)

        return {str(name) for name in names}

    for attr_name in (
        "field_names",
        "observation_names",
        "observed_series_names",
        "data_names",
        "covariate_names",
        "index_names",
    ):
        value = getattr(data, attr_name, None)

        if value is None:
            continue

        if callable(value):
            value = value()

        return {str(name) for name in value}

    fields = getattr(data, "fields", None)

    if fields is not None:
        names = {
            str(getattr(field, "name"))
            for field in fields
            if getattr(field, "name", None) is not None
        }
        if names:
            return names

    observations = getattr(data, "observations", None)

    if observations is None:
        return set()

    if isinstance(observations, Mapping):
        return {str(name) for name in observations}

    names: set[str] = set()

    for observation in observations:
        name = getattr(observation, "name", None)

        if name is not None:
            names.add(str(name))

    return names


def _compile_metadata(
    spec: Any,
    options: CompileOptions,
) -> dict[str, str]:
    metadata: dict[str, str] = {}

    spec_metadata = getattr(spec, "metadata", None)

    if spec_metadata:
        metadata.update(
            {
                str(key): str(value)
                for key, value in dict(spec_metadata).items()
            }
        )

    metadata.update(
        {str(key): str(value) for key, value in dict(options.metadata).items()}
    )

    metadata.setdefault("compiled", "true")

    return metadata


def _get_sequence_attr(
    obj: Any,
    attr_names: Iterable[str],
) -> tuple[Any, ...]:
    for attr_name in attr_names:
        value = getattr(obj, attr_name, None)

        if value is None:
            continue

        if callable(value):
            value = value()

        return tuple(value)

    return tuple()


def _dependency_set(
    obj: Any,
    attr_name: str,
) -> set[str]:
    attr = getattr(obj, attr_name, None)

    if attr is None:
        return set()

    if callable(attr):
        value = attr()
    else:
        value = attr

    if value is None:
        return set()

    return {str(item) for item in value}


def _name_of(obj: Any) -> str:
    name = getattr(obj, "name", None)

    if name is None:
        raise CompileError(
            f"Expected object {obj!r} to expose a 'name' attribute."
        )

    return str(name)


def _duplicates(values: Sequence[str]) -> list[str]:
    return sorted({value for value in values if values.count(value) > 1})


__all__ = [
    "CompileError",
    "CompileOptions",
    "compile_model",
    "compile_model_from_dict",
]
