from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any, Literal

import jax.numpy as jnp
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from dynode.typing import DynodeName


class InitializerValueSpec(BaseModel):
    """
    Base class for values used to construct initial conditions.

    These values are intentionally declarative and serializable.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )

    type: str

    def dependencies(self) -> set[str]:
        """
        Return parameter names needed to evaluate this value.
        """
        raise NotImplementedError

    def data_dependencies(self) -> set[str]:
        """
        Return observed-data names needed to evaluate this value.
        """
        return set()

    def evaluate(
        self,
        context: dict[str, Any] | None = None,
        data: Any | None = None,
    ) -> Any:
        raise NotImplementedError


class ConstantInitialValueSpec(InitializerValueSpec):
    """
    Literal numeric initial value.

    Examples
    --------
    0.0
    [100, 200, 300]
    [[10, 20], [30, 40]]
    """

    type: Literal["constant"] = "constant"

    value: Any = Field(
        description="Scalar or nested numeric list used as an initial value."
    )

    @model_validator(mode="after")
    def validate_numeric_tree(self) -> Self:
        self._validate_numeric_value(self.value)
        return self

    def dependencies(self) -> set[str]:
        return set()

    def evaluate(
        self,
        context: dict[str, Any] | None = None,
        data: Any | None = None,
    ) -> Any:
        if isinstance(self.value, list):
            return jnp.asarray(self.value)

        return self.value

    @classmethod
    def _validate_numeric_value(cls, value: Any) -> None:
        if isinstance(value, bool):
            return

        if isinstance(value, (int, float)):
            return

        if isinstance(value, list):
            for item in value:
                cls._validate_numeric_value(item)

            # Catch ragged nested lists early.
            try:
                array = np.asarray(value)
            except Exception as exc:
                raise ValueError("Constant initial value is not array-like.") from exc

            if array.dtype == object:
                raise ValueError(
                    "Constant initial value appears to be ragged. "
                    "Use a rectangular nested list."
                )

            return

        raise TypeError(
            "Constant initial values must be numeric scalars or nested numeric lists. "
            f"Got {type(value).__name__}."
        )


class ParamInitialValueSpec(InitializerValueSpec):
    """
    Reference to a sampled or resolved parameter.
    """

    type: Literal["param_ref"] = "param_ref"

    name: str

    def dependencies(self) -> set[str]:
        return {self.name}

    def evaluate(
        self,
        context: dict[str, Any] | None = None,
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


class DeterministicInitialValueSpec(InitializerValueSpec):
    """
    Reference to an already evaluated deterministic parameter.
    """

    type: Literal["deterministic_ref"] = "deterministic_ref"

    name: str

    def dependencies(self) -> set[str]:
        return {self.name}

    def evaluate(
        self,
        context: dict[str, Any] | None = None,
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


class DataInitialValueSpec(InitializerValueSpec):
    """
    Reference to observed data.

    This is useful when initial conditions are derived from the first observed
    population size, first observed prevalence, imported data arrays, etc.
    """

    type: Literal["data_ref"] = "data_ref"

    name: str = Field(
        description="Name of the data field or observed series to use."
    )

    index: int | None = Field(
        default=None,
        description=(
            "Optional index into the referenced data array. "
            "For example, index=0 uses the first observation."
        ),
    )

    def dependencies(self) -> set[str]:
        return set()

    def data_dependencies(self) -> set[str]:
        return {self.name}

    def evaluate(
        self,
        context: dict[str, Any] | None = None,
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
        """
        Supports:
        - DataSpec with get_observation(...)
        - dict returned by DataSpec.as_jax()
        - plain dicts
        """
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

        raise KeyError(
            f"Could not find data reference {self.name!r}."
        )


InitializerValue = Annotated[
    ConstantInitialValueSpec
    | ParamInitialValueSpec
    | DeterministicInitialValueSpec
    | DataInitialValueSpec,
    Field(discriminator="type"),
]


def as_initializer_value(value: Any) -> Any:
    """
    Convenience helper.

    Allows:

        value=0.0

    instead of:

        value={"type": "constant", "value": 0.0}
    """
    if isinstance(value, InitializerValueSpec):
        return value

    if isinstance(value, dict) and "type" in value:
        return value

    return {
        "type": "constant",
        "value": value,
    }


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
        default_factory=lambda: ConstantInitialValueSpec(value=0.0),
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
        if isinstance(data, dict):
            data = dict(data)

            if "value" in data:
                data["value"] = as_initializer_value(data["value"])

        return data

    @property
    def dependencies(self) -> set[str]:
        return self.value.dependencies()

    @property
    def data_dependencies(self) -> set[str]:
        return self.value.data_dependencies()

    def evaluate(
        self,
        context: dict[str, Any] | None = None,
        data: Any | None = None,
    ) -> Any:
        return self.value.evaluate(context=context, data=data)


class InitializerSpec(BaseModel):
    """
    Declarative specification for constructing an initial ODE state.

    This class validates and builds initial values for compartments. It should
    not sample parameters or call NumPyro primitives.

    The runtime layer can call:

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
        description="Initializer strategy. Currently supports explicit compartment values.",
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
            compartment.compartment_name
            for compartment in self.compartments
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

        This is called by SimulationSpec.validate_initializer_compatible().
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

        ModelSpec can call this if initializer validation eventually needs
        access to parameters, data, or transmission settings.
        """
        self.validate_against_simulation(model.simulation)

        parameter_names = getattr(model.parameters, "resolved_parameter_names", set())
        if callable(parameter_names):
            parameter_names = parameter_names()

        missing_parameter_refs = sorted(self.dependencies - set(parameter_names))

        if missing_parameter_refs:
            raise ValueError(
                "Initializer refers to unknown parameters: "
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

    def _validate_constant_shapes_against_simulation(self, simulation: Any) -> None:
        """
        Validate shapes that are knowable at config-validation time.

        Parameter/data references may only be shape-checkable at runtime.
        """
        for compartment_name, init in self.compartment_map.items():
            if not isinstance(init.value, ConstantInitialValueSpec):
                continue

            target_shape = tuple(simulation.compartment_shape(compartment_name))
            value = init.value.evaluate()
            value_shape = tuple(jnp.asarray(value).shape)

            if self._shape_matches(value_shape, target_shape):
                continue

            if init.allow_broadcast and self._can_broadcast(value_shape, target_shape):
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
                        f"Missing initial condition for compartment {compartment_name!r}."
                    )

                raw_value = 0.0
                allow_broadcast = True
            else:
                raw_value = init.evaluate(context=context, data=data)
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

        This is useful if your ODE runtime stores the full state as one vector.
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

        return tuple(broadcast_shape) == target_shape 
