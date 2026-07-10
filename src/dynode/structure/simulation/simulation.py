from __future__ import annotations

from functools import cached_property
from types import SimpleNamespace

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from dynode.structure.bins.age import AgeBin
from dynode.structure.bins.base import BinSpec
from dynode.structure.compartments.compartment import CompartmentSpec
from dynode.structure.dimensions.base import DimensionSpec
from dynode.structure.initializer.initializer import InitializerSpec

from .flattening import (
    dimensions_by_name,
    flatten_bins,
    flatten_dims,
    flatten_unique_bins,
    flatten_unique_dims,
)
from .indexing import build_simulation_idx
from .shapes import (
    compartment_size,
    compartment_sizes,
    compartment_slices,
    total_state_size,
)
from .validation import (
    validate_initializer_compatible,
    validate_shared_dimensions_are_identical,
    validate_unique_compartment_names,
    validate_unique_dimension_names_within_compartments,
)


class SimulationSpec(BaseModel):
    """
    Declarative simulation structure for a dynamic ODE model.
    """

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
        validate_assignment=True,
    )

    initializer: InitializerSpec = Field(
        description="Specification for constructing the initial ODE state."
    )

    compartments: tuple[CompartmentSpec, ...] = Field(
        min_length=1,
        description="Compartments included in the ODE state.",
    )

    @property
    def compartment_names(self) -> list[str]:
        return [compartment.name for compartment in self.compartments]

    @property
    def compartments_to_idx(self) -> dict[str, int]:
        return {
            compartment.name: i
            for i, compartment in enumerate(self.compartments)
        }

    @model_validator(mode="after")
    def validate_unique_compartment_names(self) -> Self:
        validate_unique_compartment_names(self)
        return self

    @model_validator(mode="after")
    def validate_unique_dimension_names_within_compartments(self) -> Self:
        validate_unique_dimension_names_within_compartments(self)
        return self

    @model_validator(mode="after")
    def validate_shared_dimensions_are_identical(self) -> Self:
        validate_shared_dimensions_are_identical(self)
        return self

    @model_validator(mode="after")
    def validate_initializer_compatible(self) -> Self:
        validate_initializer_compatible(self)
        return self

    def get_compartment(self, name: str) -> CompartmentSpec:
        for compartment in self.compartments:
            if compartment.name == name:
                return compartment

        raise KeyError(
            f"Unknown compartment {name!r}. "
            f"Known compartments are: {self.compartment_names}."
        )

    def has_compartment(self, name: str) -> bool:
        return name in self.compartments_to_idx

    def flatten_dims(self) -> list[DimensionSpec]:
        return flatten_dims(self)

    def flatten_unique_dims(self) -> list[DimensionSpec]:
        return flatten_unique_dims(self)

    def flatten_bins(self) -> list[BinSpec]:
        return flatten_bins(self)

    def flatten_unique_bins(self) -> list[BinSpec]:
        return flatten_unique_bins(self)

    def dimensions_by_name(self) -> dict[str, DimensionSpec]:
        return dimensions_by_name(self)

    def get_dimension(self, name: str) -> DimensionSpec:
        dimensions = self.dimensions_by_name()

        try:
            return dimensions[name]
        except KeyError as exc:
            raise KeyError(
                f"Unknown dimension {name!r}. "
                f"Known dimensions are: {list(dimensions)}."
            ) from exc

    def has_dimension(self, name: str) -> bool:
        return name in self.dimensions_by_name()

    def age_dimensions(self) -> list[DimensionSpec]:
        age_dimensions: list[DimensionSpec] = []

        for dimension in self.flatten_unique_dims():
            if dimension.bins and all(
                isinstance(bin_, AgeBin) for bin_ in dimension.bins
            ):
                age_dimensions.append(dimension)

        return age_dimensions

    def get_age_bins(self) -> list[AgeBin]:
        age_dimensions = self.age_dimensions()

        if not age_dimensions:
            return []

        return list(age_dimensions[0].bins)

    def compartment_shape(self, compartment_name: str) -> tuple[int, ...]:
        return self.get_compartment(compartment_name).shape

    def compartment_shapes(self) -> dict[str, tuple[int, ...]]:
        return {
            compartment.name: compartment.shape
            for compartment in self.compartments
        }

    def compartment_size(self, compartment_name: str) -> int:
        return compartment_size(self, compartment_name)

    def compartment_sizes(self) -> dict[str, int]:
        return compartment_sizes(self)

    @property
    def total_state_size(self) -> int:
        return total_state_size(self)

    @property
    def compartment_slices(self) -> dict[str, slice]:
        return compartment_slices(self)

    @cached_property
    def idx(self) -> SimpleNamespace:
        return build_simulation_idx(self)
