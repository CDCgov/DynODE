from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Mapping

import jax.numpy as jnp

from .aliases import ArrayLike
from .parameter_layout import RuntimeParameterLayout
from .state_layout import StateLayout
from .transmission import RuntimeTransmission
from .utils import readonly_mapping


@dataclass(frozen=True, slots=True)
class RuntimeModel:
    """
    Compiled runtime model.

    This is the object that runtime modules should share.

    It contains:
    - original ModelSpec
    - compiled state layout
    - compiled parameter layout
    - compiled transmission layout

    It does not:
    - sample parameters
    - call numpyro.sample
    - build y0
    - call diffrax.diffeqsolve
    """

    spec: Any
    state_layout: StateLayout
    parameter_layout: RuntimeParameterLayout
    transmission: RuntimeTransmission
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "metadata",
            readonly_mapping(self.metadata),
        )

    @classmethod
    def from_spec(cls, spec: Any) -> RuntimeModel:
        """
        Convenience constructor.

        compile_model.py can simply call this at first:

            def compile_model(spec: ModelSpec) -> RuntimeModel:
                return RuntimeModel.from_spec(spec)

        Later, compile_model.py can add caching, logging, or more advanced
        compilation steps around this.
        """
        state_layout = StateLayout.from_simulation(spec.simulation)

        parameter_layout = RuntimeParameterLayout.from_spec(spec.parameters)

        get_age_bins = getattr(spec.simulation, "get_age_bins", None)

        if callable(get_age_bins):
            age_bins = tuple(get_age_bins())
        else:
            age_bins = tuple()

        transmission = RuntimeTransmission.from_spec(
            spec.transmission,
            age_bins=age_bins,
        )

        return cls(
            spec=spec,
            state_layout=state_layout,
            parameter_layout=parameter_layout,
            transmission=transmission,
            metadata=getattr(spec, "metadata", {}),
        )

    @property
    def name(self) -> str:
        return str(getattr(self.spec, "name", ""))

    @property
    def version(self) -> str | None:
        return getattr(self.spec, "version", None)

    @property
    def simulation_spec(self) -> Any:
        return self.spec.simulation

    @property
    def parameter_spec(self) -> Any:
        return self.spec.parameters

    @property
    def solver_spec(self) -> Any:
        return self.spec.solver

    @property
    def initializer_spec(self) -> Any:
        return self.spec.simulation.initializer

    @property
    def data_spec(self) -> Any | None:
        return getattr(self.spec, "data", None)

    @property
    def total_state_size(self) -> int:
        return self.state_layout.total_size

    @property
    def compartment_names(self) -> tuple[str, ...]:
        return self.state_layout.compartment_names

    @property
    def strain_names(self) -> tuple[str, ...]:
        return self.transmission.strain_names

    @property
    def idx(self) -> SimpleNamespace:
        return self.state_layout.idx

    def zeros_state_flat(self, dtype: Any = float) -> Any:
        return self.state_layout.zeros_flat(dtype=dtype)

    def zeros_state_dict(self, dtype: Any = float) -> dict[str, Any]:
        return self.state_layout.zeros_dict(dtype=dtype)

    def flatten_state(
        self,
        state: Mapping[str, ArrayLike],
        *,
        allow_broadcast: bool = False,
    ) -> Any:
        return self.state_layout.flatten(
            state,
            allow_broadcast=allow_broadcast,
        )

    def unflatten_state(self, flat_state: ArrayLike) -> dict[str, Any]:
        return self.state_layout.unflatten(flat_state)

    def state_view(
        self,
        flat_state: ArrayLike,
        compartment_name: str,
    ) -> Any:
        return self.state_layout.view(
            flat_state,
            compartment_name,
        )

    def replace_state_view(
        self,
        flat_state: ArrayLike,
        compartment_name: str,
        value: ArrayLike,
        *,
        allow_broadcast: bool = False,
    ) -> Any:
        return self.state_layout.replace(
            flat_state,
            compartment_name,
            value,
            allow_broadcast=allow_broadcast,
        )

    def evaluate_interaction_matrix(
        self,
        context: Mapping[str, Any] | None = None,
        data: Any | None = None,
    ) -> Any:
        return self.transmission.evaluate_interaction_matrix(
            context=context,
            data=data,
        )

    def introduction_age_mask_matrix(self, dtype: Any = jnp.int32) -> Any:
        return self.transmission.introduction_age_mask_matrix(dtype=dtype)

    def validate_parameter_context(
        self,
        context: Mapping[str, Any],
        *,
        require_all: bool = True,
    ) -> None:
        self.parameter_layout.validate_context(
            context,
            require_all=require_all,
        )

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "n_compartments": self.state_layout.n_compartments,
            "compartment_names": self.compartment_names,
            "total_state_size": self.total_state_size,
            "n_strains": self.transmission.n_strains,
            "strain_names": self.strain_names,
            "prior_names": self.parameter_layout.prior_names,
            "deterministic_names": self.parameter_layout.deterministic_names,
            "has_data": self.data_spec is not None,
            "has_introductions": self.transmission.has_introductions,
        }
