from __future__ import annotations

from typing import Any, Literal

import jax.numpy as jnp
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from dynode.value.constant import ConstantValueSpec

from .initial_condition import CompartmentInitialConditionSpec
from .shapes import can_broadcast, coerce_to_shape, shape_matches


class InitializerSpec(BaseModel):
    """
    Declarative specification for constructing an initial ODE state.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )

    type: Literal["explicit"] = Field(
        default="explicit",
        description=(
            "Initializer strategy. Currently supports explicit compartment values."
        ),
    )

    compartments: tuple[CompartmentInitialConditionSpec, ...] = Field(
        default_factory=tuple,
        description="Initial conditions by compartment.",
    )

    missing_compartment_policy: Literal["zero", "error"] = Field(
        default="zero",
        description=(
            "What to do when a compartment is not explicitly initialized. "
            "'zero' fills missing compartments with zeros; 'error' requires "
            "every simulation compartment to be specified."
        ),
    )

    metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Optional initializer metadata.",
    )

    @property
    def initialized_compartment_names(self) -> list[str]:
        return [
            compartment.compartment_name for compartment in self.compartments
        ]

    @property
    def compartment_map(self) -> dict[str, CompartmentInitialConditionSpec]:
        return {
            compartment.compartment_name: compartment
            for compartment in self.compartments
        }

    @property
    def dependencies(self) -> set[str]:
        deps: set[str] = set()

        for compartment in self.compartments:
            deps |= compartment.dependencies

        return deps

    @property
    def parameter_dependencies(self) -> set[str]:
        deps: set[str] = set()

        for compartment in self.compartments:
            deps |= compartment.parameter_dependencies

        return deps

    @property
    def deterministic_dependencies(self) -> set[str]:
        deps: set[str] = set()

        for compartment in self.compartments:
            deps |= compartment.deterministic_dependencies

        return deps

    @property
    def data_dependencies(self) -> set[str]:
        deps: set[str] = set()

        for compartment in self.compartments:
            deps |= compartment.data_dependencies

        return deps

    @model_validator(mode="after")
    def validate_initializer(self) -> Self:
        self._validate_unique_compartment_initializers()
        return self

    def _validate_unique_compartment_initializers(self) -> None:
        names = self.initialized_compartment_names
        duplicates = sorted({name for name in names if names.count(name) > 1})

        if duplicates:
            raise ValueError(
                "Each compartment can only have one initializer. "
                f"Duplicates: {duplicates}."
            )

    def validate_against_simulation(self, simulation: Any) -> None:
        simulation_names = set(simulation.compartment_names)
        initializer_names = set(self.initialized_compartment_names)

        unknown = sorted(initializer_names - simulation_names)

        if unknown:
            raise ValueError(
                "Initializer refers to compartments not present in the simulation: "
                f"{unknown}. Known compartments are: {simulation.compartment_names}."
            )

        missing = sorted(simulation_names - initializer_names)

        if missing and self.missing_compartment_policy == "error":
            raise ValueError(
                "Initializer is missing initial conditions for compartments: "
                f"{missing}."
            )

        self._validate_constant_shapes_against_simulation(simulation)

    def validate_against_model(self, model: Any) -> None:
        self.validate_against_simulation(model.simulation)

        resolved_parameter_names = getattr(
            model,
            "available_parameter_names",
            None,
        )

        if resolved_parameter_names is None:
            resolved_parameter_names = getattr(
                model.parameters,
                "resolved_parameter_names",
                set(),
            )

        if callable(resolved_parameter_names):
            resolved_parameter_names = resolved_parameter_names()

        missing_parameter_refs = sorted(
            self.dependencies - set(resolved_parameter_names)
        )

        if missing_parameter_refs:
            raise ValueError(
                "Initializer refers to unknown parameters or deterministic values: "
                f"{missing_parameter_refs}."
            )

        if model.data is not None:
            available_data = set(getattr(model.data, "observation_names", []))
            missing_data_refs = sorted(self.data_dependencies - available_data)

            if missing_data_refs:
                raise ValueError(
                    "Initializer refers to unknown data series: "
                    f"{missing_data_refs}."
                )

    def _validate_constant_shapes_against_simulation(
        self, simulation: Any
    ) -> None:
        for compartment_name, init in self.compartment_map.items():
            if not isinstance(init.value, ConstantValueSpec):
                continue

            target_shape = tuple(
                simulation.compartment_shape(compartment_name)
            )
            value = init.value.evaluate()
            value_shape = tuple(jnp.asarray(value).shape)

            if shape_matches(value_shape, target_shape):
                continue

            if init.allow_broadcast and can_broadcast(
                value_shape,
                target_shape,
            ):
                continue

            raise ValueError(
                f"Initial value for compartment {compartment_name!r} has shape "
                f"{value_shape}, but expected {target_shape}. "
                f"allow_broadcast={init.allow_broadcast}."
            )

    def build_dict(
        self,
        simulation: Any,
        context: dict[str, Any] | None = None,
        data: Any | None = None,
    ) -> dict[str, Any]:
        self.validate_against_simulation(simulation)

        state: dict[str, Any] = {}
        init_by_compartment = self.compartment_map

        for compartment in simulation.compartments:
            compartment_name = compartment.name
            target_shape = tuple(compartment.shape)

            init = init_by_compartment.get(compartment_name)

            if init is None:
                if self.missing_compartment_policy == "error":
                    raise ValueError(
                        f"Missing initial condition for compartment "
                        f"{compartment_name!r}."
                    )

                raw_value = 0.0
                allow_broadcast = True
            else:
                raw_value = init.evaluate(
                    context=context,
                    data=data,
                )
                allow_broadcast = init.allow_broadcast

            state[compartment_name] = coerce_to_shape(
                raw_value,
                target_shape=target_shape,
                compartment_name=compartment_name,
                allow_broadcast=allow_broadcast,
            )

        return state

    def build_flat(
        self,
        simulation: Any,
        context: dict[str, Any] | None = None,
        data: Any | None = None,
    ):
        state_dict = self.build_dict(
            simulation=simulation,
            context=context,
            data=data,
        )

        pieces = [
            jnp.ravel(state_dict[compartment.name])
            for compartment in simulation.compartments
        ]

        if not pieces:
            return jnp.asarray([])

        return jnp.concatenate(pieces)

    def build(
        self,
        simulation: Any,
        context: dict[str, Any] | None = None,
        data: Any | None = None,
        flat: bool = False,
    ):
        if flat:
            return self.build_flat(
                simulation=simulation,
                context=context,
                data=data,
            )

        return self.build_dict(
            simulation=simulation,
            context=context,
            data=data,
        )
