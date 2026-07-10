from __future__ import annotations

from math import prod


def compartment_size(simulation, compartment_name: str) -> int:
    shape = simulation.compartment_shape(compartment_name)
    return prod(shape) if shape else 1


def compartment_sizes(simulation) -> dict[str, int]:
    return {
        compartment.name: compartment_size(simulation, compartment.name)
        for compartment in simulation.compartments
    }


def total_state_size(simulation) -> int:
    return sum(compartment_sizes(simulation).values())


def compartment_slices(simulation) -> dict[str, slice]:
    slices: dict[str, slice] = {}
    start = 0

    for compartment in simulation.compartments:
        size = compartment_size(simulation, compartment.name)
        stop = start + size
        slices[compartment.name] = slice(start, stop)
        start = stop

    return slices
