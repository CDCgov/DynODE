from __future__ import annotations

from typing import ClassVar, Literal

from .discretized import DiscretizedPositiveIntBin


class AgeBin(DiscretizedPositiveIntBin):
    """
    Age bin with inclusive minimum and maximum ages.
    """

    type: Literal["age"] = "age"

    default_name_prefix: ClassVar[str] = "a"

    @classmethod
    def default_name(
        cls,
        min_value: int,
        max_value: int,
    ) -> str:
        return f"a{min_value}_{max_value}"

    @property
    def min_age(self) -> int:
        return self.min_value

    @property
    def max_age(self) -> int:
        return self.max_value

    @property
    def label(self) -> str:
        return f"{self.min_value}-{self.max_value}"

    def contains_age(self, age: int) -> bool:
        return self.contains(age)
