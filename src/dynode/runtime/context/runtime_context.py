from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from dynode.runtime.layout.runtime_model import RuntimeModel


@dataclass(frozen=True)
class RuntimeContext:
    """Context passed across state construction, RHS evaluation, and observation."""

    runtime: RuntimeModel
    params: Mapping[str, Any]
    data: Any | None = None
    scope: str | None = None
    shared_params: Mapping[str, Any] = field(default_factory=dict)
    static: Mapping[str, Any] = field(default_factory=dict)
    extra: Mapping[str, Any] = field(default_factory=dict)

    def merged_params(self) -> dict[str, Any]:
        merged = dict(self.shared_params)
        merged.update(dict(self.params))
        return merged

    def get(self, name: str, default: Any = None) -> Any:
        if name in self.params:
            return self.params[name]
        if name in self.shared_params:
            return self.shared_params[name]
        if name in self.static:
            return self.static[name]
        if (
            self.data is not None
            and isinstance(self.data, Mapping)
            and name in self.data
        ):
            return self.data[name]
        return default
