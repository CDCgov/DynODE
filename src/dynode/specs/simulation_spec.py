from functools import cached_property
from math import prod
from types import SimpleNamespace
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from . import CompartmentSpec, InitializerSpec
from .bin_spec import AgeBin, BinSpec
from .dimension_spec import DimensionSpec


class _IntWithAttributes(int):
    """
    Integer index that also exposes named attributes.

    Example
    -------
    idx.S.age.adult

    where:
    - idx.S is the compartment index
    - idx.S.age is the dimension index
    - idx.S.age.adult is the bin index
    """

    def __new__(cls, value: int, **attributes: Any):
        obj = super().__new__(cls, value)
        for key, val in attributes.items():
            setattr(obj, key, val)
        return obj

    def __repr__(self) -> str:
        return f"{int(self)} {self.__dict__}"

    def __str__(self) -> str:
        return str(int(self))


class SimulationSpec(BaseModel):
    """
    Declarative simulation structure for a dynamic ODE model.

    This class owns:
    - compartments
    - dimensions
    - bin layout
    - initializer spec
    - static shape/index helpers

    It should not own:
    - priors
    - NumPyro sampling
    - strain/transmission validation
    - observed-data likelihoods
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
        names = self.compartment_names
        duplicates = sorted({name for name in names if names.count(name) > 1})

        if duplicates:
            raise ValueError(
                f"Compartment names must be unique. Duplicates: {duplicates}."
            )

        return self

    @model_validator(mode="after")
    def validate_unique_dimension_names_within_compartments(self) -> Self:
        for compartment in self.compartments:
            dimension_names = [
                dimension.name for dimension in compartment.dimensions
            ]
            duplicates = sorted(
                {
                    name
                    for name in dimension_names
                    if dimension_names.count(name) > 1
                }
            )

            if duplicates:
                raise ValueError(
                    f"Compartment {compartment.name!r} contains duplicate "
                    f"dimension names: {duplicates}."
                )

        return self

    @model_validator(mode="after")
    def validate_shared_dimensions_are_identical(self) -> Self:
        """
        If two compartments use a dimension with the same name, those dimensions
        should mean the same thing.

        For example, if both S and I have an 'age' dimension, they should use
        the same age bins in the same order.
        """
        dimension_by_name: dict[str, DimensionSpec] = {}

        for dimension in self.flatten_dims():
            previous = dimension_by_name.get(dimension.name)

            if previous is None:
                dimension_by_name[dimension.name] = dimension
                continue

            if dimension != previous:
                raise ValueError(
                    f"Dimension {dimension.name!r} has inconsistent definitions "
                    "across compartments. If these are intended to be different, "
                    "give them different names."
                )

        return self

    @model_validator(mode="after")
    def validate_initializer_compatible(self) -> Self:
        """
        Let InitializerSpec optionally validate itself against the simulation.

        This keeps SimulationSpec generic while still allowing specialized
        initializers to check required compartments, shapes, dimensions, etc.
        """
        validate = getattr(
            self.initializer,
            "validate_against_simulation",
            None,
        )

        if callable(validate):
            validate(self)

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
        """
        Flatten all compartment dimensions while preserving order.

        Order:
        1. compartment order
        2. dimension order within each compartment
        """
        flattened: list[DimensionSpec] = []

        for compartment in self.compartments:
            flattened.extend(compartment.dimensions)

        return flattened

    def flatten_unique_dims(self) -> list[DimensionSpec]:
        """
        Return unique dimensions by name, preserving first-seen order.
        """
        seen: set[str] = set()
        unique: list[DimensionSpec] = []

        for dimension in self.flatten_dims():
            if dimension.name not in seen:
                seen.add(dimension.name)
                unique.append(dimension)

        return unique

    def flatten_bins(self) -> list[BinSpec]:
        """
        Flatten all bins across all compartments and dimensions.

        This preserves compartment, dimension, and bin order.
        """
        flattened: list[BinSpec] = []

        for dimension in self.flatten_dims():
            flattened.extend(dimension.bins)

        return flattened

    def flatten_unique_bins(self) -> list[BinSpec]:
        """
        Return unique bins preserving first-seen order.

        Useful for validation, not for constructing the ODE state.
        """
        unique: list[BinSpec] = []

        for bin_ in self.flatten_bins():
            if bin_ not in unique:
                unique.append(bin_)

        return unique

    def dimensions_by_name(self) -> dict[str, DimensionSpec]:
        return {
            dimension.name: dimension
            for dimension in self.flatten_unique_dims()
        }

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
        """
        Return dimensions whose bins are AgeBin instances.
        """
        age_dimensions: list[DimensionSpec] = []

        for dimension in self.flatten_unique_dims():
            if dimension.bins and all(
                isinstance(bin_, AgeBin) for bin_ in dimension.bins
            ):
                age_dimensions.append(dimension)

        return age_dimensions

    def get_age_bins(self) -> list[AgeBin]:
        """
        Return the first age-bin structure found in the model.

        If your framework supports multiple age dimensions, you may want to
        replace this with a stricter method that requires a dimension name.
        """
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
        shape = self.compartment_shape(compartment_name)
        return prod(shape) if shape else 1

    def compartment_sizes(self) -> dict[str, int]:
        return {
            compartment.name: self.compartment_size(compartment.name)
            for compartment in self.compartments
        }

    @property
    def total_state_size(self) -> int:
        """
        Total flattened size of the ODE state.

        Useful if your runtime stores the full state as one flat JAX array.
        """
        return sum(self.compartment_sizes().values())

    @property
    def compartment_slices(self) -> dict[str, slice]:
        """
        Slices for each compartment in a flattened state vector.

        Example
        -------
        y[simulation.compartment_slices["infectious"]]
        """
        slices: dict[str, slice] = {}
        start = 0

        for compartment in self.compartments:
            size = self.compartment_size(compartment.name)
            stop = start + size
            slices[compartment.name] = slice(start, stop)
            start = stop

        return slices

    @cached_property
    def idx(self) -> SimpleNamespace:
        """
        Enum-like static index helper.

        Example
        -------
        simulation.idx.S
        simulation.idx.S.age
        simulation.idx.S.age.adult

        This is convenient for model authoring, but runtime JAX code should
        eventually receive plain integer indices, slices, arrays, or pytrees.
        """
        compartments_namespace = SimpleNamespace()

        for compartment_idx, compartment in enumerate(self.compartments):
            dimension_attrs: dict[str, Any] = {}

            for dimension_idx, dimension in enumerate(compartment.dimensions):
                bin_attrs: dict[str, int] = {}

                for bin_idx, bin_ in enumerate(dimension.bins):
                    bin_name = getattr(bin_, "name", None)

                    if bin_name is None:
                        raise ValueError(
                            f"Bin {bin_!r} in dimension {dimension.name!r} "
                            "does not expose a 'name' attribute."
                        )

                    bin_attrs[bin_name] = bin_idx

                dimension_attrs[dimension.name] = _IntWithAttributes(
                    dimension_idx,
                    **bin_attrs,
                )

            setattr(
                compartments_namespace,
                compartment.name,
                _IntWithAttributes(
                    compartment_idx,
                    **dimension_attrs,
                ),
            )

        return compartments_namespace
