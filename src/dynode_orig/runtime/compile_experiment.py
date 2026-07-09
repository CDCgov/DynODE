from __future__ import annotations

from typing import Any

from dynode.specs.experiment_spec import ExperimentSpec

from .compile_model import CompileOptions, compile_model
from .experiment_runtime import ExperimentRuntime, ModelInstanceRuntime
from .runtime_model import RuntimeParameterLayout


class ExperimentCompileError(ValueError):
    pass


def compile_parameter_block(block: Any) -> RuntimeParameterLayout:
    return RuntimeParameterLayout.from_spec(block)


def compile_experiment(
    spec: ExperimentSpec,
    *,
    options: CompileOptions | None = None,
) -> ExperimentRuntime:
    options = options or CompileOptions()
    shared_layout = compile_parameter_block(spec.shared_parameters)
    instances: list[ModelInstanceRuntime] = []
    shared_names = set(shared_layout.resolved_name_set)

    for instance in spec.instances:
        instance_layout = compile_parameter_block(instance.parameters)
        available_external = _available_instance_names(
            shared_names=shared_names,
            instance_layout=instance_layout,
            static_context=instance.static_context,
        )
        missing_external = sorted(
            set(instance.model.external_parameter_names) - available_external
        )
        if missing_external:
            raise ExperimentCompileError(
                f"Instance {instance.key!r} model requires unavailable external parameters: {missing_external}."
            )

        model_options = CompileOptions(
            run_spec_validation_hooks=options.run_spec_validation_hooks,
            validate_runtime_layout=options.validate_runtime_layout,
            validate_parameter_dependencies=options.validate_parameter_dependencies,
            validate_data_dependencies=options.validate_data_dependencies,
            validate_transmission_dependencies=options.validate_transmission_dependencies,
            validate_initializer_dependencies=options.validate_initializer_dependencies,
            validate_data_spec=options.validate_data_spec,
            allow_prior_dependencies_on_deterministics=options.allow_prior_dependencies_on_deterministics,
            metadata=options.metadata,
            external_parameter_names=frozenset(available_external),
        )
        runtime = compile_model(instance.model, options=model_options)
        instances.append(
            ModelInstanceRuntime(
                key=instance.key,
                runtime=runtime,
                parameter_layout=instance_layout,
                data_spec=instance.data or instance.model.data,
                static_context=instance.static_context,
                t0=instance.t0,
                t1=instance.t1,
                metadata=instance.metadata,
            )
        )

    return ExperimentRuntime(
        name=spec.name,
        shared_parameter_layout=shared_layout,
        instances=tuple(instances),
        spec=spec,
        metadata=spec.metadata,
    )


def _available_instance_names(
    *,
    shared_names: set[str],
    instance_layout: RuntimeParameterLayout,
    static_context: dict[str, Any],
) -> set[str]:
    """Return names available while compiling one model instance."""
    available = set(shared_names)
    available |= set(instance_layout.resolved_name_set)
    available |= {str(name) for name in static_context}

    parameter_context = static_context.get("parameter_context")
    if isinstance(parameter_context, dict):
        available |= {str(name) for name in parameter_context}

    return available
