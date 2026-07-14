from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from dynode.runtime.layout.runtime_model import (
    RuntimeModel,
    RuntimeParameterLayout,
)


@dataclass(frozen=True)
class ModelInstanceRuntime:
    key: str
    runtime: RuntimeModel
    parameter_layout: RuntimeParameterLayout
    data_spec: Any | None = None
    static_context: Mapping[str, Any] = field(default_factory=dict)
    t0: float | None = None
    t1: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
