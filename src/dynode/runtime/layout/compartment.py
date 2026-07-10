from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Mapping

import jax.numpy as jnp

from .aliases import ArrayLike
from .dimension import RuntimeDimension
from .utils import IntWithAttributes, duplicates, readonly_mapping, shape_size


@dataclass(frozen=True, slots=True)
class RuntimeCompartment:
    """
    Compiled runtime representation of one compartment.

    Stores the compartment's position in the flat ODE state vector.
    """

    name: str
    index: int
    start: int
    stop: int
    shape: tuple[int, ...]
    dimensions: tuple[RuntimeDimension, ...] = field(default_factory=tuple)
    spec: Any | None = field(default=None, repr=False, compare=False)

    dimensions_to_axis: Mapping[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError(
                f"RuntimeCompartment.index must be non-negative. Got {self.index}."
            )

        if self.start < 0:
            raise ValueError(
                f"RuntimeCompartment.start must be non-negative. Got {self.start}."
            )

        if self.stop <= self.start:
            raise ValueError(
                f"RuntimeCompartment {self.name!r} has invalid slice "
                f"start={self.start}, stop={self.stop}."
            )

        expected_size = shape_size(self.shape)
        actual_size = self.stop - self.start

        if actual_size != expected_size:
            raise ValueError(
                f"RuntimeCompartment {self.name!r} slice has size={actual_size}, "
                f"but shape {self.shape} implies size={expected_size}."
            )

        if len(self.shape) != len(self.dimensions):
            raise ValueError(
                f"RuntimeCompartment {self.name!r} has shape with "
                f"{len(self.shape)} axes but {len(self.dimensions)} dimensions."
            )

        for expected_axis, dimension in enumerate(self.dimensions):
            if dimension.axis != expected_axis:
                raise ValueError(
                    f"RuntimeCompartment {self.name!r} dimension {dimension.name!r} "
                    f"has axis={dimension.axis}, expected axis={expected_axis}."
                )

            if dimension.size != self.shape[expected_axis]:
                raise ValueError(
                    f"RuntimeCompartment {self.name!r} dimension {dimension.name!r} "
                    f"has size={dimension.size}, but shape axis {expected_axis} "
                    f"has size={self.shape[expected_axis]}."
                )

        dimension_names = self.dimension_names
        duplicate_names = duplicates(dimension_names)
        if duplicate_names:
            raise ValueError(
                f"RuntimeCompartment {self.name!r} has duplicate dimensions: "
                f"{duplicate_names}."
            )

        object.__setattr__(
            self,
            "dimensions_to_axis",
            readonly_mapping(
                {
                    dimension.name: dimension.axis
                    for dimension in self.dimensions
                }
            ),
        )

    @classmethod
    def from_spec(
        cls,
        compartment: Any,
        *,
        index: int,
        start: int,
    ) -> RuntimeCompartment:
        dimensions = tuple(
            RuntimeDimension.from_spec(dimension, axis=axis)
            for axis, dimension in enumerate(compartment.dimensions)
        )

        shape = tuple(dimension.size for dimension in dimensions)
        size = shape_size(shape)

        return cls(
            name=str(compartment.name),
            index=index,
            start=start,
            stop=start + size,
            shape=shape,
            dimensions=dimensions,
            spec=compartment,
        )

    @property
    def state_slice(self) -> slice:
        return slice(self.start, self.stop)

    @property
    def size(self) -> int:
        return self.stop - self.start

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def is_scalar(self) -> bool:
        return self.ndim == 0

    @property
    def is_stratified(self) -> bool:
        return self.ndim > 0

    @property
    def dimension_names(self) -> tuple[str, ...]:
        return tuple(dimension.name for dimension in self.dimensions)

    @property
    def idx(self) -> SimpleNamespace:
        namespace = SimpleNamespace()

        for dimension in self.dimensions:
            bin_attrs = {
                bin_name: bin_idx
                for bin_name, bin_idx in dimension.bins_to_idx.items()
            }

            setattr(
                namespace,
                dimension.name,
                IntWithAttributes(
                    dimension.axis,
                    **bin_attrs,
                ),
            )

        return namespace

    def has_dimension(self, name: str) -> bool:
        return name in self.dimensions_to_axis

    def axis_of(self, dimension_name: str) -> int:
        try:
            return self.dimensions_to_axis[dimension_name]
        except KeyError as exc:
            raise KeyError(
                f"Unknown dimension {dimension_name!r} in compartment "
                f"{self.name!r}. Known dimensions are: {self.dimension_names}."
            ) from exc

    def get_dimension(self, name: str) -> RuntimeDimension:
        axis = self.axis_of(name)
        return self.dimensions[axis]

    def zeros(self, dtype: Any = float) -> Any:
        return jnp.zeros(self.shape, dtype=dtype)

    def flatten_value(
        self,
        value: ArrayLike,
        *,
        allow_broadcast: bool = False,
    ) -> Any:
        array = jnp.asarray(value)
        value_shape = tuple(array.shape)

        if value_shape == self.shape:
            return jnp.ravel(array)

        if allow_broadcast:
            try:
                return jnp.ravel(jnp.broadcast_to(array, self.shape))
            except ValueError as exc:
                raise ValueError(
                    f"Value for compartment {self.name!r} has shape {value_shape}, "
                    f"which cannot be broadcast to {self.shape}."
                ) from exc

        raise ValueError(
            f"Value for compartment {self.name!r} has shape {value_shape}, "
            f"but expected {self.shape}."
        )

    def unflatten_from(self, flat_state: ArrayLike) -> Any:
        flat_state = jnp.asarray(flat_state)

        if flat_state.ndim != 1:
            raise ValueError(
                f"Expected flat state vector with ndim=1. Got shape {flat_state.shape}."
            )

        if flat_state.shape[0] < self.stop:
            raise ValueError(
                f"Flat state vector has length {flat_state.shape[0]}, but "
                f"compartment {self.name!r} requires stop index {self.stop}."
            )

        return jnp.reshape(flat_state[self.state_slice], self.shape)
