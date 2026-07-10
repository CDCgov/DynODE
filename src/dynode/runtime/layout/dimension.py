from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Mapping

from .utils import duplicates, name_of, readonly_mapping


@dataclass(frozen=True, slots=True)
class RuntimeDimension:
    """
    Compiled runtime representation of one dimension within one compartment.

    This object is built from DimensionSpec but contains only static runtime
    layout information.
    """

    name: str
    axis: int
    size: int
    bin_names: tuple[str, ...]
    bin_specs: tuple[Any, ...] = field(default_factory=tuple, repr=False)

    bins_to_idx: Mapping[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.axis < 0:
            raise ValueError(
                f"RuntimeDimension.axis must be non-negative. Got {self.axis}."
            )

        if self.size <= 0:
            raise ValueError(
                f"RuntimeDimension.size must be positive. Got {self.size}."
            )

        if len(self.bin_names) != self.size:
            raise ValueError(
                f"RuntimeDimension {self.name!r} has size={self.size}, "
                f"but len(bin_names)={len(self.bin_names)}."
            )

        duplicate_names = duplicates(self.bin_names)
        if duplicate_names:
            raise ValueError(
                f"RuntimeDimension {self.name!r} has duplicate bin names: "
                f"{duplicate_names}."
            )

        if self.bin_specs and len(self.bin_specs) != self.size:
            raise ValueError(
                f"RuntimeDimension {self.name!r} has {len(self.bin_specs)} bin specs, "
                f"but size={self.size}."
            )

        object.__setattr__(
            self,
            "bins_to_idx",
            readonly_mapping(
                {name: i for i, name in enumerate(self.bin_names)}
            ),
        )

    @classmethod
    def from_spec(
        cls,
        dimension: Any,
        *,
        axis: int,
    ) -> RuntimeDimension:
        bins = tuple(dimension.bins)
        bin_names = tuple(name_of(bin_) for bin_ in bins)

        return cls(
            name=str(dimension.name),
            axis=axis,
            size=len(bins),
            bin_names=bin_names,
            bin_specs=bins,
        )

    @property
    def idx(self) -> SimpleNamespace:
        namespace = SimpleNamespace()

        for bin_name, bin_idx in self.bins_to_idx.items():
            setattr(namespace, bin_name, bin_idx)

        return namespace

    def has_bin(self, name: str) -> bool:
        return name in self.bins_to_idx

    def index_of(self, bin_name: str) -> int:
        try:
            return self.bins_to_idx[bin_name]
        except KeyError as exc:
            raise KeyError(
                f"Unknown bin {bin_name!r} in dimension {self.name!r}. "
                f"Known bins are: {self.bin_names}."
            ) from exc

    def get_bin_spec(self, bin_name: str) -> Any:
        if not self.bin_specs:
            raise ValueError(
                f"RuntimeDimension {self.name!r} does not store bin specs."
            )

        return self.bin_specs[self.index_of(bin_name)]
