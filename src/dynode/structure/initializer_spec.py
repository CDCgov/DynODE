from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import jax.numpy as jnp
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from dynode.typing import DynodeName
from dynode.value.coercion import as_value_spec
from dynode.value.constant import ConstantValueSpec

if TYPE_CHECKING:
    from dynode.value.unions import InitializerValue


class CompartmentInitialConditionSpec(BaseModel):
    """
    Initial condition for one compartment.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )

    compartment_name: DynodeName = Field(
        description="Name of the compartment to initialize."
    )

    value: InitializerValue = Field(
        default_factory=lambda: ConstantValueSpec(value=0.0),
        description=(
            "Initial value for this compartment. Scalars are broadcast to the "
            "full compartment shape."
        ),
    )

    allow_broadcast: bool = Field(
        default=True,
        description="Whether scalar or lower-dimensional values may be broadcast.",
    )

    description: str | None = Field(
        default=None,
        description="Optional human-readable description.",
    )

    @model_validator(mode="before")
    @classmethod
    def coerce_value(cls, data: Any) -> Any:
        if isinstance(data, dict) and "value" in data:
            data = dict(data)
            data["value"] = as_value_spec(data["value"])

        return data

    @property
    def dependencies(self) -> set[str]:
        """
        Parameter and deterministic dependencies required to evaluate this
        initial condition.
        """
        return self.value.dependencies()

    @property
    def parameter_dependencies(self) -> set[str]:
        return self.value.parameter_dependencies()

    @property
    def deterministic_dependencies(self) -> set[str]:
        return self.value.deterministic_dependencies()

    @property
    def data_dependencies(self) -> set[str]:
        return self.value.data_dependencies()

    def evaluate(
        self,
        context: dict[str, Any] | None = None,
        data: Any | None = None,
    ) -> Any:
        return self.value.evaluate(
            context=context,
            data=data,
        )


class InitializerSpec(BaseModel):
    """
    Declarative specification for constructing an initial ODE state.

    This class validates and builds initial values for compartments. It should
    not sample parameters or call NumPyro primitives.

    Runtime code can call:

        initializer.build_dict(simulation, context, data)
        initializer.build_flat(simulation, context, data)
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
        """
        All parameter-like dependencies required by the initializer.
        Includes sampled parameter refs and deterministic refs.
        """
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
        """
        Validate this initializer against SimulationSpec.
        """
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
        """
        Optional full-model validation hook.

        ModelSpec can call this when initializer validation needs access to
        parameters, data, or transmission settings.
        """
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
        """
        Validate shapes that are knowable at config-validation time.

        ParamRef, DeterministicRef, DataRef, and expression values may only be
        shape-checkable at runtime.
        """
        for compartment_name, init in self.compartment_map.items():
            if not isinstance(init.value, ConstantValueSpec):
                continue

            target_shape = tuple(
                simulation.compartment_shape(compartment_name)
            )
            value = init.value.evaluate()
            value_shape = tuple(jnp.asarray(value).shape)

            if self._shape_matches(value_shape, target_shape):
                continue

            if init.allow_broadcast and self._can_broadcast(
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
        """
        Build initial state as a dictionary of compartment arrays.

        Returns
        -------
        dict[str, jax.Array]
            Mapping from compartment name to array with the compartment shape.
        """
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

            state[compartment_name] = self._coerce_to_shape(
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
        """
        Build initial state as a flattened JAX vector.

        Useful if your ODE runtime stores the full state as one vector.
        """
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
        """
        Convenience method.

        Parameters
        ----------
        flat:
            If True, return a single flattened JAX vector.
            If False, return a dict of compartment arrays.
        """
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

    @staticmethod
    def _coerce_to_shape(
        value: Any,
        target_shape: tuple[int, ...],
        compartment_name: str,
        allow_broadcast: bool,
    ):
        array = jnp.asarray(value)
        value_shape = tuple(array.shape)

        if value_shape == target_shape:
            return array

        if allow_broadcast:
            try:
                return jnp.broadcast_to(array, target_shape)
            except ValueError as exc:
                raise ValueError(
                    f"Initial value for compartment {compartment_name!r} has shape "
                    f"{value_shape}, which cannot be broadcast to {target_shape}."
                ) from exc

        raise ValueError(
            f"Initial value for compartment {compartment_name!r} has shape "
            f"{value_shape}, but expected {target_shape}."
        )

    @staticmethod
    def _shape_matches(
        value_shape: tuple[int, ...],
        target_shape: tuple[int, ...],
    ) -> bool:
        return value_shape == target_shape

    @staticmethod
    def _can_broadcast(
        value_shape: tuple[int, ...],
        target_shape: tuple[int, ...],
    ) -> bool:
        try:
            broadcast_shape = np.broadcast_shapes(value_shape, target_shape)
        except ValueError:
            return False

        return tuple(broadcast_shape) == target_shape
