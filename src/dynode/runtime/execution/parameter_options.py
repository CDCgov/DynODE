from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class ParameterSamplingOptions:
    """
    Options controlling parameter sampling and deterministic resolution.
    """

    validate_dependencies: bool = True

    record_deterministics: bool = True

    allow_initial_context_overwrite: bool = False

    scope: str | None = None

    metadata: Mapping[str, str] = field(default_factory=dict)