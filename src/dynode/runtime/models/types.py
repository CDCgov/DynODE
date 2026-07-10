from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

RHSFn = Callable[..., Any]
ObserveFn = Callable[..., Any]
StateTransformFn = Callable[..., Any]
ParameterContext = Mapping[str, Any]
ModelData = Any
ExperimentData = Mapping[str, Any] | None
