from __future__ import annotations

from dynode.structure.bins.base import BinSpec
from dynode.structure.dimensions.base import DimensionSpec


def flatten_dims(simulation) -> list[DimensionSpec]:
    flattened: list[DimensionSpec] = []

    for compartment in simulation.compartments:
        flattened.extend(compartment.dimensions)

    return flattened


def flatten_unique_dims(simulation) -> list[DimensionSpec]:
    seen: set[str] = set()
    unique: list[DimensionSpec] = []

    for dimension in flatten_dims(simulation):
        if dimension.name not in seen:
            seen.add(dimension.name)
            unique.append(dimension)

    return unique


def flatten_bins(simulation) -> list[BinSpec]:
    flattened: list[BinSpec] = []

    for dimension in flatten_dims(simulation):
        flattened.extend(dimension.bins)

    return flattened


def flatten_unique_bins(simulation) -> list[BinSpec]:
    unique: list[BinSpec] = []

    for bin_ in flatten_bins(simulation):
        if bin_ not in unique:
            unique.append(bin_)

    return unique


def dimensions_by_name(simulation) -> dict[str, DimensionSpec]:
    return {
        dimension.name: dimension
        for dimension in flatten_unique_dims(simulation)
    }
