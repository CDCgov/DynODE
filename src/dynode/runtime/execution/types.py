from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable, Literal

ParameterContext = dict[str, Any]
ParameterMapping = Mapping[str, Any]
FlatState = Any
StateDict = dict[str, Any]
StateMapping = Mapping[str, Any]
RHSFn = Callable[..., Any]

RHSStateFormat = Literal["flat", "dict"]
RHSCallStyle = Literal["auto", "standard", "keyword"]
