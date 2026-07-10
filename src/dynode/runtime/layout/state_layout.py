from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Mapping

import jax.numpy as jnp

from .aliases import ArrayLike
from .compartment import RuntimeCompartment
from .utils import IntWithAttributes, duplicates, readonly_mapping


@dataclass(frozen=True, slots=True)
class StateLayout:
    """
    Compiled layout of the full ODE state vector.

    This is the central object used by state_builder.py, ode_solver.py, and RHS
    code to move between flat vectors and named compartment arrays.
    """

    compartments: tuple[RuntimeCompartment, ...]

    total_size: int = field(init=False)
    compartments_to_idx: Mapping[str, int] = field(init=False, repr=False)
    compartment_slices: Mapping[str, slice] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.compartments:
            raise ValueError("StateLayout requires at least one compartment.")

        names = self.compartment_names
        duplicate_names = duplicates(names)

        if duplicate_names:
            raise ValueError(
                f"StateLayout has duplicate compartment names: {duplicate_names}."
            )

        expected_start = 0

        for expected_index, compartment in enumerate(self.compartments):
            if compartment.index != expected_index:
                raise ValueError(
                    f"Compartment {compartment.name!r} has index={compartment.index}, "
                    f"expected index={expected_index}."
                )

            if compartment.start != expected_start:
                raise ValueError(
                    f"Compartment {compartment.name!r} has start={compartment.start}, "
                    f"expected start={expected_start}."
                )

            expected_start = compartment.stop

        object.__setattr__(self, "total_size", expected_start)

        object.__setattr__(
            self,
            "compartments_to_idx",
            readonly_mapping(
                {
                    compartment.name: compartment.index
                    for compartment in self.compartments
                }
            ),
        )

        object.__setattr__(
            self,
            "compartment_slices",
            readonly_mapping(
                {
                    compartment.name: compartment.state_slice
                    for compartment in self.compartments
                }
            ),
        )

    @classmethod
    def from_simulation(cls, simulation: Any) -> StateLayout:
        compartments: list[RuntimeCompartment] = []
        start = 0

        for index, compartment_spec in enumerate(simulation.compartments):
            runtime_compartment = RuntimeCompartment.from_spec(
                compartment_spec,
                index=index,
                start=start,
            )

            compartments.append(runtime_compartment)
            start = runtime_compartment.stop

        return cls(compartments=tuple(compartments))

    @property
    def n_compartments(self) -> int:
        return len(self.compartments)

    @property
    def compartment_names(self) -> tuple[str, ...]:
        return tuple(compartment.name for compartment in self.compartments)

    @property
    def idx(self) -> SimpleNamespace:
        namespace = SimpleNamespace()

        for compartment in self.compartments:
            dimension_attrs: dict[str, Any] = {}

            for dimension in compartment.dimensions:
                bin_attrs = {
                    bin_name: bin_idx
                    for bin_name, bin_idx in dimension.bins_to_idx.items()
                }

                dimension_attrs[dimension.name] = IntWithAttributes(
                    dimension.axis,
                    **bin_attrs,
                )

            setattr(
                namespace,
                compartment.name,
                IntWithAttributes(
                    compartment.index,
                    **dimension_attrs,
                ),
            )

        return namespace

    def has_compartment(self, name: str) -> bool:
        return name in self.compartments_to_idx

    def get_compartment(self, name: str) -> RuntimeCompartment:
        try:
            return self.compartments[self.compartments_to_idx[name]]
        except KeyError as exc:
            raise KeyError(
                f"Unknown compartment {name!r}. "
                f"Known compartments are: {self.compartment_names}."
            ) from exc

    def slice_of(self, compartment_name: str) -> slice:
        return self.get_compartment(compartment_name).state_slice

    def shape_of(self, compartment_name: str) -> tuple[int, ...]:
        return self.get_compartment(compartment_name).shape

    def size_of(self, compartment_name: str) -> int:
        return self.get_compartment(compartment_name).size

    def validate_flat_state(self, flat_state: ArrayLike) -> None:
        flat_state = jnp.asarray(flat_state)

        if flat_state.ndim != 1:
            raise ValueError(
                f"Expected flat state vector with ndim=1. Got shape {flat_state.shape}."
            )

        if flat_state.shape[0] != self.total_size:
            raise ValueError(
                f"Expected flat state vector of length {self.total_size}. "
                f"Got length {flat_state.shape[0]}."
            )

    def zeros_flat(self, dtype: Any = float) -> Any:
        return jnp.zeros((self.total_size,), dtype=dtype)

    def zeros_dict(self, dtype: Any = float) -> dict[str, Any]:
        return {
            compartment.name: compartment.zeros(dtype=dtype)
            for compartment in self.compartments
        }

    def flatten(
        self,
        state: Mapping[str, ArrayLike],
        *,
        allow_broadcast: bool = False,
    ) -> Any:
        pieces: list[Any] = []

        for compartment in self.compartments:
            if compartment.name not in state:
                raise KeyError(
                    f"State mapping is missing compartment {compartment.name!r}."
                )

            pieces.append(
                compartment.flatten_value(
                    state[compartment.name],
                    allow_broadcast=allow_broadcast,
                )
            )

        return jnp.concatenate(pieces)

    def unflatten(self, flat_state: ArrayLike) -> dict[str, Any]:
        self.validate_flat_state(flat_state)

        return {
            compartment.name: compartment.unflatten_from(flat_state)
            for compartment in self.compartments
        }

    def view(
        self,
        flat_state: ArrayLike,
        compartment_name: str,
    ) -> Any:
        self.validate_flat_state(flat_state)
        return self.get_compartment(compartment_name).unflatten_from(
            flat_state
        )

    def replace(
        self,
        flat_state: ArrayLike,
        compartment_name: str,
        value: ArrayLike,
        *,
        allow_broadcast: bool = False,
    ) -> Any:
        self.validate_flat_state(flat_state)

        compartment = self.get_compartment(compartment_name)
        flat_value = compartment.flatten_value(
            value,
            allow_broadcast=allow_broadcast,
        )

        return (
            jnp.asarray(flat_state).at[compartment.state_slice].set(flat_value)
        )
