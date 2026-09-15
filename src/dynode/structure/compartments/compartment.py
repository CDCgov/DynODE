from __future__ import annotations

from math import prod
from types import SimpleNamespace

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from dynode.structure.bins.base import BinSpec
from dynode.structure.dimensions.unions import AnyDimensionSpec
from dynode.structure.indexing import IntWithAttributes
from dynode.typing import DynodeName


class CompartmentSpec(BaseModel):
    """
    Declarative specification for one ODE compartment.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )

    name: DynodeName = Field(
        description="Compartment name. Must be unique within a SimulationSpec.",
    )

    dimensions: tuple[AnyDimensionSpec, ...] = Field(
        default_factory=tuple,
        description=(
            "Dimension definitions for this compartment. "
            "An empty tuple means the compartment is unstratified."
        ),
    )

    description: str | None = Field(
        default=None,
        description="Optional human-readable description of the compartment.",
    )

    metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Optional metadata for documentation, UI display, or auditing.",
    )

    @model_validator(mode="after")
    def validate_compartment(self) -> Self:
        self._validate_unique_dimension_names()
        return self

    @property
    def dimension_names(self) -> list[str]:
        return [dimension.name for dimension in self.dimensions]

    @property
    def dimensions_to_idx(self) -> dict[str, int]:
        return {
            dimension.name: i for i, dimension in enumerate(self.dimensions)
        }

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(len(dimension) for dimension in self.dimensions)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        return prod(self.shape) if self.shape else 1

    @property
    def is_stratified(self) -> bool:
        return bool(self.dimensions)

    @property
    def is_scalar(self) -> bool:
        return not self.dimensions

    @property
    def idx(self) -> SimpleNamespace:
        dims_namespace = SimpleNamespace()

        for dimension_idx, dimension in enumerate(self.dimensions):
            bin_attrs = self._bin_index_attrs(dimension)

            setattr(
                dims_namespace,
                dimension.name,
                IntWithAttributes(
                    dimension_idx,
                    **bin_attrs,
                ),
            )

        return dims_namespace

    def get_dimension(self, name: str) -> AnyDimensionSpec:
        for dimension in self.dimensions:
            if dimension.name == name:
                return dimension

        raise KeyError(
            f"Unknown dimension {name!r} in compartment {self.name!r}. "
            f"Known dimensions are: {self.dimension_names}."
        )

    def has_dimension(self, name: str) -> bool:
        return name in self.dimensions_to_idx

    def axis_of(self, dimension_name: str) -> int:
        try:
            return self.dimensions_to_idx[dimension_name]
        except KeyError as exc:
            raise KeyError(
                f"Unknown dimension {dimension_name!r} in compartment "
                f"{self.name!r}. Known dimensions are: {self.dimension_names}."
            ) from exc

    def bins_for_dimension(self, dimension_name: str) -> tuple[BinSpec, ...]:
        return tuple(self.get_dimension(dimension_name).bins)

    def dimension_size(self, dimension_name: str) -> int:
        return len(self.get_dimension(dimension_name))

    def flatten_bins(self) -> list[BinSpec]:
        bins: list[BinSpec] = []

        for dimension in self.dimensions:
            bins.extend(dimension.bins)

        return bins

    def same_layout_as(
        self,
        other: CompartmentSpec,
        *,
        require_same_name: bool = False,
    ) -> bool:
        if require_same_name and self.name != other.name:
            return False

        return self.dimensions == other.dimensions

    def _validate_unique_dimension_names(self) -> None:
        names = self.dimension_names
        duplicates = sorted({name for name in names if names.count(name) > 1})

        if duplicates:
            raise ValueError(
                f"Compartment {self.name!r} contains duplicate dimension "
                f"names: {duplicates}."
            )

    @staticmethod
    def _bin_index_attrs(dimension: AnyDimensionSpec) -> dict[str, int]:
        attrs: dict[str, int] = {}

        for bin_idx, bin_ in enumerate(dimension.bins):
            bin_name = getattr(bin_, "name", None)

            if bin_name is None:
                raise ValueError(
                    f"Bin {bin_!r} in dimension {dimension.name!r} does not "
                    "expose a 'name' attribute."
                )

            attrs[bin_name] = bin_idx

        return attrs


CompartmentSpec.model_rebuild(
    _types_namespace={
        "AnyDimensionSpec": AnyDimensionSpec,
    }
)
