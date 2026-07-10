from __future__ import annotations

from typing import Any, Iterable

from pydantic import BaseModel


def walk_references(
    obj: Any,
    *,
    path: str,
) -> Iterable[tuple[str, str, str]]:
    if obj is None:
        return

    kind = reference_kind(obj)

    if kind is not None:
        name = getattr(obj, "name", None)
        if name is not None:
            yield (path, kind, str(name))
        return

    if isinstance(obj, BaseModel):
        for field_name in obj.model_fields:
            yield from walk_references(
                getattr(obj, field_name), path=f"{path}.{field_name}"
            )
        return

    if isinstance(obj, dict):
        for key, value in obj.items():
            yield from walk_references(value, path=f"{path}[{key!r}]")
        return

    if isinstance(obj, (list, tuple, set, frozenset)):
        for idx, value in enumerate(obj):
            yield from walk_references(value, path=f"{path}[{idx}]")
        return


def reference_kind(obj: Any) -> str | None:
    class_name = type(obj).__name__

    if class_name in {"ParamRef", "ParameterRef"}:
        return "parameter"

    if class_name == "DeterministicRef":
        return "deterministic"

    return None
