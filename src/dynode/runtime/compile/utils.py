from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from .errors import CompileError


def get_sequence_attr(
    obj: Any,
    attr_names: Iterable[str],
) -> tuple[Any, ...]:
    for attr_name in attr_names:
        value = getattr(obj, attr_name, None)

        if value is None:
            continue

        if callable(value):
            value = value()

        return tuple(value)

    return tuple()


def dependency_set(
    obj: Any,
    attr_name: str,
) -> set[str]:
    attr = getattr(obj, attr_name, None)

    if attr is None:
        return set()

    if callable(attr):
        value = attr()
    else:
        value = attr

    if value is None:
        return set()

    return {str(item) for item in value}


def name_of(obj: Any) -> str:
    name = getattr(obj, "name", None)

    if name is None:
        raise CompileError(
            f"Expected object {obj!r} to expose a 'name' attribute."
        )

    return str(name)


def duplicates(values: Sequence[str]) -> list[str]:
    return sorted({value for value in values if values.count(value) > 1})
