from __future__ import annotations

from dynode.runtime.layout.runtime_model import RuntimeModel

from .availability import available_data_names, available_parameter_names
from .errors import CompileError
from .options import CompileOptions
from .utils import dependency_set


def validate_runtime_model(
    runtime: RuntimeModel,
    *,
    options: CompileOptions,
) -> None:
    errors: list[str] = []

    validators = []

    if options.validate_runtime_layout:
        validators.append(validate_runtime_layout)

    if options.validate_parameter_dependencies:
        validators.append(validate_parameter_dependencies)

    if options.validate_initializer_dependencies:
        validators.append(validate_initializer_dependencies)

    if options.validate_data_dependencies:
        validators.append(validate_data_dependencies)

    if options.validate_transmission_dependencies:
        validators.append(validate_transmission_dependencies)

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


def validate_runtime_layout(
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


def validate_parameter_dependencies(
    runtime: RuntimeModel,
    *,
    options: CompileOptions,
) -> None:
    available = available_parameter_names(runtime, options=options)
    errors: list[str] = []

    for prior_name, prior in runtime.parameter_layout.prior_specs.items():
        deps = dependency_set(prior, "dependencies")
        missing = sorted(deps - available)

        if missing:
            errors.append(
                f"Prior {prior_name!r} depends on unknown parameters {missing}."
            )

    for (
        deterministic_name,
        deterministic,
    ) in runtime.parameter_layout.deterministic_specs.items():
        deps = dependency_set(deterministic, "dependencies")
        missing = sorted(deps - available)

        if missing:
            errors.append(
                f"Deterministic parameter {deterministic_name!r} depends on "
                f"unknown parameters {missing}."
            )

    if errors:
        raise CompileError("; ".join(errors))


def validate_initializer_dependencies(
    runtime: RuntimeModel,
    *,
    options: CompileOptions,
) -> None:
    initializer = runtime.initializer_spec
    available = available_parameter_names(runtime, options=options)

    deps = dependency_set(initializer, "dependencies")
    missing = sorted(deps - available)

    if missing:
        raise CompileError(
            "Initializer refers to unknown parameters or deterministic values: "
            f"{missing}."
        )


def validate_data_dependencies(
    runtime: RuntimeModel,
    *,
    options: CompileOptions,
) -> None:
    del options

    data = runtime.data_spec
    available_data = available_data_names(data)

    errors: list[str] = []

    initializer_data_deps = dependency_set(
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
        data_deps = dependency_set(prior, "data_dependencies")

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
        data_deps = dependency_set(deterministic, "data_dependencies")

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


def validate_transmission_dependencies(
    runtime: RuntimeModel,
    *,
    options: CompileOptions,
) -> None:
    available = available_parameter_names(runtime, options=options)
    errors: list[str] = []

    for strain_name, strain in runtime.transmission.strain_specs.items():
        deps = dependency_set(strain, "dependencies")
        missing = sorted(deps - available)

        if missing:
            errors.append(
                f"Strain {strain_name!r} depends on unknown parameters "
                f"{missing}."
            )

        data_deps = dependency_set(strain, "data_dependencies")

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
            deps = dependency_set(interaction, "dependencies")
            missing = sorted(deps - available)

            if missing:
                errors.append(
                    f"Interaction {source_name!r}->{target_name!r} depends "
                    f"on unknown parameters {missing}."
                )

            data_deps = dependency_set(interaction, "data_dependencies")

            if data_deps:
                errors.append(
                    f"Interaction {source_name!r}->{target_name!r} has data "
                    f"dependencies, which are not allowed in transmission specs: "
                    f"{sorted(data_deps)}."
                )

    if errors:
        raise CompileError("; ".join(errors))