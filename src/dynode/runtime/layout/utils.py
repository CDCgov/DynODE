from __future__ import annotations

from math import prod
from types import MappingProxyType
from typing import Any, Mapping


def readonly_mapping(mapping: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(mapping))


def duplicates(values: tuple[str, ...]) -> list[str]:
    return sorted({value for value in values if values.count(value) > 1})


def name_of(value: Any) -> str:
    name = getattr(value, "name", None)

    if name is None:
        raise ValueError(
            f"Object {value!r} does not expose a 'name' attribute."
        )

    return str(name)


def shape_size(shape: tuple[int, ...]) -> int:
    return prod(shape) if shape else 1


class IntWithAttributes(int):
    """
    Integer index that can also expose named attributes.

    Used for ergonomic runtime indexes like:

        runtime.idx.I.age.adult
        runtime.idx.I.hist.none
    """

    def __new__(cls, value: int, **attributes: Any):
        obj = super().__new__(cls, value)

        for key, val in attributes.items():
            setattr(obj, key, val)

        return obj

    def __repr__(self) -> str:
        if self.__dict__:
            return f"{int(self)} {self.__dict__}"
        return str(int(self))

    def __str__(self) -> str:
        return str(int(self))
