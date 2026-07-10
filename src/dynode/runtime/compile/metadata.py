from __future__ import annotations

from typing import Any

from .options import CompileOptions


def compile_metadata(
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
