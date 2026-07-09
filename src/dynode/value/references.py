from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from .base import ValueSpec


class ParamRef(ValueSpec):
    """
    Reference to a sampled parameter.

    Example
    -------
    {"type": "param_ref", "name": "r0"}
    """

    type: Literal["param_ref"] = "param_ref"

    name: str
    scope: str | None = None

    def parameter_dependencies(self) -> set[str]:
        return {self.name}

    def evaluate(
        self,
        context: Mapping[str, Any] | None = None,
        data: Any | None = None,
    ) -> Any:
        if context is None:
            raise ValueError(
                f"Cannot resolve parameter reference {self.name!r} without context."
            )

        try:
            return context[self.name]
        except KeyError as exc:
            raise KeyError(
                f"Parameter {self.name!r} was not found in context. "
                f"Available values are: {sorted(context)}."
            ) from exc


class DeterministicRef(ValueSpec):
    """
    Reference to an already-resolved deterministic parameter.
    """

    type: Literal["deterministic_ref"] = "deterministic_ref"

    name: str

    def deterministic_dependencies(self) -> set[str]:
        return {self.name}

    def evaluate(
        self,
        context: Mapping[str, Any] | None = None,
        data: Any | None = None,
    ) -> Any:
        if context is None:
            raise ValueError(
                f"Cannot resolve deterministic reference {self.name!r} without context."
            )

        try:
            return context[self.name]
        except KeyError as exc:
            raise KeyError(
                f"Deterministic parameter {self.name!r} was not found in context. "
                f"Available values are: {sorted(context)}."
            ) from exc


class DataRef(ValueSpec):
    """
    Reference to observed data.

    Useful for initializers or observation-related calculations.
    """

    type: Literal["data_ref"] = "data_ref"

    name: str
    index: int | None = None

    def data_dependencies(self) -> set[str]:
        return {self.name}

    def evaluate(
        self,
        context: Mapping[str, Any] | None = None,
        data: Any | None = None,
    ) -> Any:
        if data is None:
            raise ValueError(
                f"Cannot resolve data reference {self.name!r} without data."
            )

        value = self._lookup_data_value(data)

        if self.index is not None:
            value = value[self.index]

        return value

    def _lookup_data_value(self, data: Any) -> Any:
        if hasattr(data, "get_observation"):
            observation = data.get_observation(self.name)

            if hasattr(observation, "as_jax"):
                return observation.as_jax()

            return observation.values

        if isinstance(data, Mapping):
            if self.name in data:
                return data[self.name]

            observations = data.get("observations")
            if isinstance(observations, Mapping) and self.name in observations:
                return observations[self.name]

        raise KeyError(f"Could not find data reference {self.name!r}.")
