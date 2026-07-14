from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from dynode.runtime.layout.runtime_model import RuntimeParameterLayout

from .data_bundle import DataBundle
from .model_instance_runtime import ModelInstanceRuntime


@dataclass(frozen=True)
class ExperimentRuntime:
    name: str
    shared_parameter_layout: RuntimeParameterLayout
    instances: tuple[ModelInstanceRuntime, ...]
    spec: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def instance_keys(self) -> tuple[str, ...]:
        return tuple(instance.key for instance in self.instances)

    def get_instance(self, key: str) -> ModelInstanceRuntime:
        for instance in self.instances:
            if instance.key == key:
                return instance
        raise KeyError(
            f"Unknown experiment instance {key!r}. Known instances are: {self.instance_keys}."
        )

    def data_bundle(
        self, key: str, data: Mapping[str, Any] | None = None
    ) -> DataBundle | None:
        instance = self.get_instance(key)
        if data is None:
            return (
                None
                if instance.data_spec is None
                else DataBundle(values={}, spec=instance.data_spec)
            )
        if key in data and isinstance(data[key], Mapping):
            payload = data[key]
        else:
            payload = data
        if isinstance(payload, DataBundle):
            return payload
        values = (
            payload.get("values", payload)
            if isinstance(payload, Mapping)
            else payload
        )
        metadata = (
            payload.get("metadata", {}) if isinstance(payload, Mapping) else {}
        )
        return DataBundle(
            values=values, spec=instance.data_spec, metadata=metadata
        )
