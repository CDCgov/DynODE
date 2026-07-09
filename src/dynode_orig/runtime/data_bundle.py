from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp
import numpy as np

from dynode.specs.data_spec import DataSpec


@dataclass(frozen=True)
class DataBundle(Mapping[str, Any]):
    """Runtime values paired with a DataSpec contract."""

    values: Mapping[str, Any] = field(default_factory=dict)
    spec: DataSpec | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.spec is not None:
            self.spec.validate_values(self.values)

    def __getitem__(self, key: str) -> Any:
        if key in self.values:
            return self.values[key]
        if key in self.metadata:
            return self.metadata[key]
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        yield from self.values
        for key in self.metadata:
            if key not in self.values:
                yield key

    def __len__(self) -> int:
        return len(set(self.values) | set(self.metadata))

    def get_observation(self, name: str) -> Any:
        return self.values[name]

    @property
    def observation_names(self) -> list[str]:
        if self.spec is None:
            return list(self.values)
        return self.spec.data_names

    def as_jax_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in self.values.items():
            if value is None:
                result[key] = None
            elif isinstance(value, (str, bytes)):
                result[key] = value
            else:
                try:
                    result[key] = jnp.asarray(value)
                except Exception:
                    result[key] = value
        result.update(dict(self.metadata))
        return result

    def as_numpy_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in self.values.items():
            if value is None:
                result[key] = None
            else:
                try:
                    result[key] = np.asarray(value)
                except Exception:
                    result[key] = value
        result.update(dict(self.metadata))
        return result
