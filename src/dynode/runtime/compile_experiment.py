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
        model_options = options
        runtime = compile_model(instance.model, options=model_options)
        instance_layout = compile_parameter_block(instance.parameters)
        missing_external = sorted(
            set(instance.model.external_parameter_names)
            - (
                shared_names
                | instance_layout.resolved_name_set
                | set(instance.static_context)
            )
        )
        if missing_external:
            raise ExperimentCompileError(
                f"Instance {instance.key!r} model requires unavailable external parameters: {missing_external}."
            )
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
