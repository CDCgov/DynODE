from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from dynode.structure.indexing import IntWithAttributes


def build_simulation_idx(simulation) -> SimpleNamespace:
    compartments_namespace = SimpleNamespace()

    for compartment_idx, compartment in enumerate(simulation.compartments):
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

            dimension_attrs[dimension.name] = IntWithAttributes(
                dimension_idx,
                **bin_attrs,
            )

        setattr(
            compartments_namespace,
            compartment.name,
            IntWithAttributes(
                compartment_idx,
                **dimension_attrs,
            ),
        )

    return compartments_namespace
