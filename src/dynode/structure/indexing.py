from __future__ import annotations

from typing import Any


class IntWithAttributes(int):
    """
    Integer index that can also expose named attributes.

    Used by structure specs for ergonomic static index helpers, for example:

        simulation.idx.S.age.adult
        compartment.idx.age.adult
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
