from __future__ import annotations

from dataclasses import dataclass, field
from math import prod
from types import MappingProxyType, SimpleNamespace
from typing import Any, Mapping

import jax.numpy as jnp

ArrayLike = Any


def _readonly_mapping(mapping: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(mapping))


def _duplicates(values: tuple[str, ...]) -> list[str]:
    return sorted({value for value in values if values.count(value) > 1})


def _name_of(value: Any) -> str:
    name = getattr(value, "name", None)

    if name is None:
        raise ValueError(
            f"Object {value!r} does not expose a 'name' attribute."
        )

    return str(name)


def _shape_size(shape: tuple[int, ...]) -> int:
    return prod(shape) if shape else 1


class _IntWithAttributes(int):
    """
    Integer index that can also expose named attributes.

    Used for ergonomic runtime indexes like:

        runtime.idx.I.age.adult
        runtime.idx.I.hist.none
    """

    def __new__(cls, value: int, **attributes: Any):
        obj = super().__new__(cls, value)

        for key, val in attributes.items():
            setattr(obj, key, val)

        return obj

    def __repr__(self) -> str:
        if self.__dict__:
            return f"{int(self)} {self.__dict__}"
        return str(int(self))

    def __str__(self) -> str:
        return str(int(self))


@dataclass(frozen=True, slots=True)
class RuntimeDimension:
    """
    Compiled runtime representation of one dimension within one compartment.

    This object is built from DimensionSpec but contains only static runtime
    layout information.
    """

    name: str
    axis: int
    size: int
    bin_names: tuple[str, ...]
    bin_specs: tuple[Any, ...] = field(default_factory=tuple, repr=False)

    bins_to_idx: Mapping[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.axis < 0:
            raise ValueError(
                f"RuntimeDimension.axis must be non-negative. Got {self.axis}."
            )

        if self.size <= 0:
            raise ValueError(
                f"RuntimeDimension.size must be positive. Got {self.size}."
            )

        if len(self.bin_names) != self.size:
            raise ValueError(
                f"RuntimeDimension {self.name!r} has size={self.size}, "
                f"but len(bin_names)={len(self.bin_names)}."
            )

        duplicates = _duplicates(self.bin_names)
        if duplicates:
            raise ValueError(
                f"RuntimeDimension {self.name!r} has duplicate bin names: {duplicates}."
            )

        if self.bin_specs and len(self.bin_specs) != self.size:
            raise ValueError(
                f"RuntimeDimension {self.name!r} has {len(self.bin_specs)} bin specs, "
                f"but size={self.size}."
            )

        object.__setattr__(
            self,
            "bins_to_idx",
            _readonly_mapping(
                {name: i for i, name in enumerate(self.bin_names)}
            ),
        )

    @classmethod
    def from_spec(
        cls,
        dimension: Any,
        *,
        axis: int,
    ) -> RuntimeDimension:
        bins = tuple(dimension.bins)
        bin_names = tuple(_name_of(bin_) for bin_ in bins)

        return cls(
            name=str(dimension.name),
            axis=axis,
            size=len(bins),
            bin_names=bin_names,
            bin_specs=bins,
        )

    @property
    def idx(self) -> SimpleNamespace:
        namespace = SimpleNamespace()

        for bin_name, bin_idx in self.bins_to_idx.items():
            setattr(namespace, bin_name, bin_idx)

        return namespace

    def has_bin(self, name: str) -> bool:
        return name in self.bins_to_idx

    def index_of(self, bin_name: str) -> int:
        try:
            return self.bins_to_idx[bin_name]
        except KeyError as exc:
            raise KeyError(
                f"Unknown bin {bin_name!r} in dimension {self.name!r}. "
                f"Known bins are: {self.bin_names}."
            ) from exc

    def get_bin_spec(self, bin_name: str) -> Any:
        if not self.bin_specs:
            raise ValueError(
                f"RuntimeDimension {self.name!r} does not store bin specs."
            )

        return self.bin_specs[self.index_of(bin_name)]


@dataclass(frozen=True, slots=True)
class RuntimeCompartment:
    """
    Compiled runtime representation of one compartment.

    Stores the compartment's position in the flat ODE state vector.
    """

    name: str
    index: int
    start: int
    stop: int
    shape: tuple[int, ...]
    dimensions: tuple[RuntimeDimension, ...] = field(default_factory=tuple)
    spec: Any | None = field(default=None, repr=False, compare=False)

    dimensions_to_axis: Mapping[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError(
                f"RuntimeCompartment.index must be non-negative. Got {self.index}."
            )

        if self.start < 0:
            raise ValueError(
                f"RuntimeCompartment.start must be non-negative. Got {self.start}."
            )

        if self.stop <= self.start:
            raise ValueError(
                f"RuntimeCompartment {self.name!r} has invalid slice "
                f"start={self.start}, stop={self.stop}."
            )

        expected_size = _shape_size(self.shape)
        actual_size = self.stop - self.start

        if actual_size != expected_size:
            raise ValueError(
                f"RuntimeCompartment {self.name!r} slice has size={actual_size}, "
                f"but shape {self.shape} implies size={expected_size}."
            )

        if len(self.shape) != len(self.dimensions):
            raise ValueError(
                f"RuntimeCompartment {self.name!r} has shape with "
                f"{len(self.shape)} axes but {len(self.dimensions)} dimensions."
            )

        for expected_axis, dimension in enumerate(self.dimensions):
            if dimension.axis != expected_axis:
                raise ValueError(
                    f"RuntimeCompartment {self.name!r} dimension {dimension.name!r} "
                    f"has axis={dimension.axis}, expected axis={expected_axis}."
                )

            if dimension.size != self.shape[expected_axis]:
                raise ValueError(
                    f"RuntimeCompartment {self.name!r} dimension {dimension.name!r} "
                    f"has size={dimension.size}, but shape axis {expected_axis} "
                    f"has size={self.shape[expected_axis]}."
                )

        dimension_names = self.dimension_names
        duplicates = _duplicates(dimension_names)
        if duplicates:
            raise ValueError(
                f"RuntimeCompartment {self.name!r} has duplicate dimensions: {duplicates}."
            )

        object.__setattr__(
            self,
            "dimensions_to_axis",
            _readonly_mapping(
                {
                    dimension.name: dimension.axis
                    for dimension in self.dimensions
                }
            ),
        )

    @classmethod
    def from_spec(
        cls,
        compartment: Any,
        *,
        index: int,
        start: int,
    ) -> RuntimeCompartment:
        dimensions = tuple(
            RuntimeDimension.from_spec(dimension, axis=axis)
            for axis, dimension in enumerate(compartment.dimensions)
        )

        shape = tuple(dimension.size for dimension in dimensions)
        size = _shape_size(shape)

        return cls(
            name=str(compartment.name),
            index=index,
            start=start,
            stop=start + size,
            shape=shape,
            dimensions=dimensions,
            spec=compartment,
        )

    @property
    def state_slice(self) -> slice:
        return slice(self.start, self.stop)

    @property
    def size(self) -> int:
        return self.stop - self.start

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def is_scalar(self) -> bool:
        return self.ndim == 0

    @property
    def is_stratified(self) -> bool:
        return self.ndim > 0

    @property
    def dimension_names(self) -> tuple[str, ...]:
        return tuple(dimension.name for dimension in self.dimensions)

    @property
    def idx(self) -> SimpleNamespace:
        namespace = SimpleNamespace()

        for dimension in self.dimensions:
            bin_attrs = {
                bin_name: bin_idx
                for bin_name, bin_idx in dimension.bins_to_idx.items()
            }

            setattr(
                namespace,
                dimension.name,
                _IntWithAttributes(
                    dimension.axis,
                    **bin_attrs,
                ),
            )

        return namespace

    def has_dimension(self, name: str) -> bool:
        return name in self.dimensions_to_axis

    def axis_of(self, dimension_name: str) -> int:
        try:
            return self.dimensions_to_axis[dimension_name]
        except KeyError as exc:
            raise KeyError(
                f"Unknown dimension {dimension_name!r} in compartment "
                f"{self.name!r}. Known dimensions are: {self.dimension_names}."
            ) from exc

    def get_dimension(self, name: str) -> RuntimeDimension:
        axis = self.axis_of(name)
        return self.dimensions[axis]

    def zeros(self, dtype: Any = float) -> Any:
        return jnp.zeros(self.shape, dtype=dtype)

    def flatten_value(
        self,
        value: ArrayLike,
        *,
        allow_broadcast: bool = False,
    ) -> Any:
        array = jnp.asarray(value)
        value_shape = tuple(array.shape)

        if value_shape == self.shape:
            return jnp.ravel(array)

        if allow_broadcast:
            try:
                return jnp.ravel(jnp.broadcast_to(array, self.shape))
            except ValueError as exc:
                raise ValueError(
                    f"Value for compartment {self.name!r} has shape {value_shape}, "
                    f"which cannot be broadcast to {self.shape}."
                ) from exc

        raise ValueError(
            f"Value for compartment {self.name!r} has shape {value_shape}, "
            f"but expected {self.shape}."
        )

    def unflatten_from(self, flat_state: ArrayLike) -> Any:
        flat_state = jnp.asarray(flat_state)

        if flat_state.ndim != 1:
            raise ValueError(
                f"Expected flat state vector with ndim=1. Got shape {flat_state.shape}."
            )

        if flat_state.shape[0] < self.stop:
            raise ValueError(
                f"Flat state vector has length {flat_state.shape[0]}, but "
                f"compartment {self.name!r} requires stop index {self.stop}."
            )

        return jnp.reshape(flat_state[self.state_slice], self.shape)


@dataclass(frozen=True, slots=True)
class StateLayout:
    """
    Compiled layout of the full ODE state vector.

    This is the central object used by state_builder.py, ode_solver.py, and RHS
    code to move between flat vectors and named compartment arrays.
    """

    compartments: tuple[RuntimeCompartment, ...]

    total_size: int = field(init=False)
    compartments_to_idx: Mapping[str, int] = field(init=False, repr=False)
    compartment_slices: Mapping[str, slice] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.compartments:
            raise ValueError("StateLayout requires at least one compartment.")

        names = self.compartment_names
        duplicates = _duplicates(names)

        if duplicates:
            raise ValueError(
                f"StateLayout has duplicate compartment names: {duplicates}."
            )

        expected_start = 0

        for expected_index, compartment in enumerate(self.compartments):
            if compartment.index != expected_index:
                raise ValueError(
                    f"Compartment {compartment.name!r} has index={compartment.index}, "
                    f"expected index={expected_index}."
                )

            if compartment.start != expected_start:
                raise ValueError(
                    f"Compartment {compartment.name!r} has start={compartment.start}, "
                    f"expected start={expected_start}."
                )

            expected_start = compartment.stop

        object.__setattr__(self, "total_size", expected_start)

        object.__setattr__(
            self,
            "compartments_to_idx",
            _readonly_mapping(
                {
                    compartment.name: compartment.index
                    for compartment in self.compartments
                }
            ),
        )

        object.__setattr__(
            self,
            "compartment_slices",
            _readonly_mapping(
                {
                    compartment.name: compartment.state_slice
                    for compartment in self.compartments
                }
            ),
        )

    @classmethod
    def from_simulation(cls, simulation: Any) -> StateLayout:
        compartments: list[RuntimeCompartment] = []
        start = 0

        for index, compartment_spec in enumerate(simulation.compartments):
            runtime_compartment = RuntimeCompartment.from_spec(
                compartment_spec,
                index=index,
                start=start,
            )

            compartments.append(runtime_compartment)
            start = runtime_compartment.stop

        return cls(compartments=tuple(compartments))

    @property
    def n_compartments(self) -> int:
        return len(self.compartments)

    @property
    def compartment_names(self) -> tuple[str, ...]:
        return tuple(compartment.name for compartment in self.compartments)

    @property
    def idx(self) -> SimpleNamespace:
        namespace = SimpleNamespace()

        for compartment in self.compartments:
            dimension_attrs: dict[str, Any] = {}

            for dimension in compartment.dimensions:
                bin_attrs = {
                    bin_name: bin_idx
                    for bin_name, bin_idx in dimension.bins_to_idx.items()
                }

                dimension_attrs[dimension.name] = _IntWithAttributes(
                    dimension.axis,
                    **bin_attrs,
                )

            setattr(
                namespace,
                compartment.name,
                _IntWithAttributes(
                    compartment.index,
                    **dimension_attrs,
                ),
            )

        return namespace

    def has_compartment(self, name: str) -> bool:
        return name in self.compartments_to_idx

    def get_compartment(self, name: str) -> RuntimeCompartment:
        try:
            return self.compartments[self.compartments_to_idx[name]]
        except KeyError as exc:
            raise KeyError(
                f"Unknown compartment {name!r}. "
                f"Known compartments are: {self.compartment_names}."
            ) from exc

    def slice_of(self, compartment_name: str) -> slice:
        return self.get_compartment(compartment_name).state_slice

    def shape_of(self, compartment_name: str) -> tuple[int, ...]:
        return self.get_compartment(compartment_name).shape

    def size_of(self, compartment_name: str) -> int:
        return self.get_compartment(compartment_name).size

    def validate_flat_state(self, flat_state: ArrayLike) -> None:
        flat_state = jnp.asarray(flat_state)

        if flat_state.ndim != 1:
            raise ValueError(
                f"Expected flat state vector with ndim=1. Got shape {flat_state.shape}."
            )

        if flat_state.shape[0] != self.total_size:
            raise ValueError(
                f"Expected flat state vector of length {self.total_size}. "
                f"Got length {flat_state.shape[0]}."
            )

    def zeros_flat(self, dtype: Any = float) -> Any:
        return jnp.zeros((self.total_size,), dtype=dtype)

    def zeros_dict(self, dtype: Any = float) -> dict[str, Any]:
        return {
            compartment.name: compartment.zeros(dtype=dtype)
            for compartment in self.compartments
        }

    def flatten(
        self,
        state: Mapping[str, ArrayLike],
        *,
        allow_broadcast: bool = False,
    ) -> Any:
        pieces: list[Any] = []

        for compartment in self.compartments:
            if compartment.name not in state:
                raise KeyError(
                    f"State mapping is missing compartment {compartment.name!r}."
                )

            pieces.append(
                compartment.flatten_value(
                    state[compartment.name],
                    allow_broadcast=allow_broadcast,
                )
            )

        return jnp.concatenate(pieces)

    def unflatten(self, flat_state: ArrayLike) -> dict[str, Any]:
        self.validate_flat_state(flat_state)

        return {
            compartment.name: compartment.unflatten_from(flat_state)
            for compartment in self.compartments
        }

    def view(
        self,
        flat_state: ArrayLike,
        compartment_name: str,
    ) -> Any:
        self.validate_flat_state(flat_state)
        return self.get_compartment(compartment_name).unflatten_from(
            flat_state
        )

    def replace(
        self,
        flat_state: ArrayLike,
        compartment_name: str,
        value: ArrayLike,
        *,
        allow_broadcast: bool = False,
    ) -> Any:
        self.validate_flat_state(flat_state)

        compartment = self.get_compartment(compartment_name)
        flat_value = compartment.flatten_value(
            value,
            allow_broadcast=allow_broadcast,
        )

        return (
            jnp.asarray(flat_state).at[compartment.state_slice].set(flat_value)
        )


@dataclass(frozen=True, slots=True)
class RuntimeParameterLayout:
    """
    Compiled parameter layout.

    This stores names and specs needed by parameter_sampling.py. It does not
    call numpyro.sample itself.
    """

    prior_names: tuple[str, ...] = field(default_factory=tuple)
    deterministic_names: tuple[str, ...] = field(default_factory=tuple)
    deterministic_order: tuple[Any, ...] = field(default_factory=tuple)

    prior_specs: Mapping[str, Any] = field(default_factory=dict, repr=False)
    deterministic_specs: Mapping[str, Any] = field(
        default_factory=dict, repr=False
    )

    prior_to_idx: Mapping[str, int] = field(init=False, repr=False)
    deterministic_to_idx: Mapping[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        duplicate_priors = _duplicates(self.prior_names)
        if duplicate_priors:
            raise ValueError(f"Duplicate prior names: {duplicate_priors}.")

        duplicate_deterministic = _duplicates(self.deterministic_names)
        if duplicate_deterministic:
            raise ValueError(
                f"Duplicate deterministic parameter names: {duplicate_deterministic}."
            )

        collisions = sorted(
            set(self.prior_names) & set(self.deterministic_names)
        )
        if collisions:
            raise ValueError(
                "A parameter cannot be both sampled and deterministic. "
                f"Colliding names: {collisions}."
            )

        ordered_names = tuple(
            _name_of(spec) for spec in self.deterministic_order
        )

        if set(ordered_names) != set(self.deterministic_names):
            raise ValueError(
                "deterministic_order must contain exactly the deterministic "
                "parameters listed in deterministic_names. "
                f"deterministic_names={self.deterministic_names}, "
                f"ordered_names={ordered_names}."
            )

        object.__setattr__(
            self,
            "prior_to_idx",
            _readonly_mapping(
                {name: i for i, name in enumerate(self.prior_names)}
            ),
        )

        object.__setattr__(
            self,
            "deterministic_to_idx",
            _readonly_mapping(
                {name: i for i, name in enumerate(self.deterministic_names)}
            ),
        )

        object.__setattr__(
            self,
            "prior_specs",
            _readonly_mapping(self.prior_specs),
        )

        object.__setattr__(
            self,
            "deterministic_specs",
            _readonly_mapping(self.deterministic_specs),
        )

    @classmethod
    def from_spec(cls, parameters: Any) -> RuntimeParameterLayout:
        priors = tuple(getattr(parameters, "priors", ()))
        deterministic = tuple(getattr(parameters, "deterministic", ()))

        deterministic_execution_order = getattr(
            parameters,
            "deterministic_execution_order",
            None,
        )

        if callable(deterministic_execution_order):
            deterministic_order = tuple(deterministic_execution_order())
        else:
            deterministic_order = deterministic

        prior_names = tuple(_name_of(prior) for prior in priors)
        deterministic_names = tuple(_name_of(spec) for spec in deterministic)

        return cls(
            prior_names=prior_names,
            deterministic_names=deterministic_names,
            deterministic_order=deterministic_order,
            prior_specs={_name_of(prior): prior for prior in priors},
            deterministic_specs={
                _name_of(spec): spec for spec in deterministic
            },
        )

    @property
    def resolved_names(self) -> tuple[str, ...]:
        return self.prior_names + self.deterministic_names

    @property
    def resolved_name_set(self) -> set[str]:
        return set(self.resolved_names)

    def has_parameter(self, name: str) -> bool:
        return name in self.resolved_name_set

    def has_prior(self, name: str) -> bool:
        return name in self.prior_to_idx

    def has_deterministic(self, name: str) -> bool:
        return name in self.deterministic_to_idx

    def get_prior(self, name: str) -> Any:
        try:
            return self.prior_specs[name]
        except KeyError as exc:
            raise KeyError(
                f"Unknown prior {name!r}. Known priors are: {self.prior_names}."
            ) from exc

    def get_deterministic(self, name: str) -> Any:
        try:
            return self.deterministic_specs[name]
        except KeyError as exc:
            raise KeyError(
                f"Unknown deterministic parameter {name!r}. "
                f"Known deterministic parameters are: {self.deterministic_names}."
            ) from exc

    def validate_context(
        self,
        context: Mapping[str, Any],
        *,
        require_all: bool = True,
    ) -> None:
        if not require_all:
            return

        missing = sorted(self.resolved_name_set - set(context))

        if missing:
            raise ValueError(
                f"Parameter context is missing required values: {missing}."
            )


@dataclass(frozen=True, slots=True)
class RuntimeTransmission:
    """
    Compiled transmission layout.

    Stores strain indexes, interaction matrix specs, and introduction age masks.
    """

    strain_names: tuple[str, ...]
    interaction_matrix_spec: tuple[tuple[Any, ...], ...]
    introduction_age_masks: Mapping[str, tuple[int, ...]] = field(
        default_factory=dict
    )
    strain_specs: Mapping[str, Any] = field(default_factory=dict, repr=False)

    strains_to_idx: Mapping[str, int] = field(init=False, repr=False)
    introduced_strain_names: tuple[str, ...] = field(init=False)
    introduced_strain_indices: tuple[int, ...] = field(init=False)
    n_age_bins_for_introduction: int = field(init=False)

    def __post_init__(self) -> None:
        if not self.strain_names:
            raise ValueError(
                "RuntimeTransmission requires at least one strain."
            )

        duplicates = _duplicates(self.strain_names)
        if duplicates:
            raise ValueError(f"Duplicate strain names: {duplicates}.")

        n_strains = len(self.strain_names)

        if len(self.interaction_matrix_spec) != n_strains:
            raise ValueError(
                "interaction_matrix_spec must have one row per strain. "
                f"Expected {n_strains}, got {len(self.interaction_matrix_spec)}."
            )

        for row in self.interaction_matrix_spec:
            if len(row) != n_strains:
                raise ValueError(
                    "interaction_matrix_spec must be square with shape "
                    f"({n_strains}, {n_strains})."
                )

        unknown_mask_names = sorted(
            set(self.introduction_age_masks) - set(self.strain_names)
        )

        if unknown_mask_names:
            raise ValueError(
                "introduction_age_masks contains unknown strains: "
                f"{unknown_mask_names}."
            )

        mask_lengths = {
            len(mask) for mask in self.introduction_age_masks.values()
        }

        if len(mask_lengths) > 1:
            raise ValueError(
                "All introduction age masks must have the same length. "
                f"Got lengths: {sorted(mask_lengths)}."
            )

        n_age_bins = next(iter(mask_lengths)) if mask_lengths else 0

        normalized_masks = {
            strain_name: tuple(
                int(value)
                for value in self.introduction_age_masks.get(
                    strain_name,
                    tuple(0 for _ in range(n_age_bins)),
                )
            )
            for strain_name in self.strain_names
        }

        introduced_names: list[str] = []

        for strain_name in self.strain_names:
            strain_spec = self.strain_specs.get(strain_name)
            is_introduced = bool(getattr(strain_spec, "is_introduced", False))

            if is_introduced or any(normalized_masks[strain_name]):
                introduced_names.append(strain_name)

        strains_to_idx = {
            strain_name: i for i, strain_name in enumerate(self.strain_names)
        }

        object.__setattr__(
            self,
            "strains_to_idx",
            _readonly_mapping(strains_to_idx),
        )

        object.__setattr__(
            self,
            "introduction_age_masks",
            _readonly_mapping(normalized_masks),
        )

        object.__setattr__(
            self,
            "strain_specs",
            _readonly_mapping(self.strain_specs),
        )

        object.__setattr__(
            self,
            "introduced_strain_names",
            tuple(introduced_names),
        )

        object.__setattr__(
            self,
            "introduced_strain_indices",
            tuple(strains_to_idx[name] for name in introduced_names),
        )

        object.__setattr__(
            self,
            "n_age_bins_for_introduction",
            n_age_bins,
        )

    @classmethod
    def from_spec(
        cls,
        transmission: Any,
        *,
        age_bins: tuple[Any, ...] = tuple(),
    ) -> RuntimeTransmission:
        strain_names_value = getattr(transmission, "strain_names", None)

        if callable(strain_names_value):
            strain_names_value = strain_names_value()

        if strain_names_value is None:
            strain_names = tuple(
                _name_of(strain) for strain in transmission.strains
            )
        else:
            strain_names = tuple(str(name) for name in strain_names_value)

        strains = tuple(getattr(transmission, "strains", ()))

        interaction_matrix = tuple(
            tuple(row) for row in transmission.interaction_matrix_spec()
        )

        masks: dict[str, tuple[int, ...]] = {}

        for strain in strains:
            strain_name = _name_of(strain)

            introduction_age_mask = getattr(
                strain, "introduction_age_mask", None
            )

            if callable(introduction_age_mask):
                masks[strain_name] = tuple(
                    int(value) for value in introduction_age_mask(age_bins)
                )
            else:
                masks[strain_name] = tuple(0 for _ in age_bins)

        return cls(
            strain_names=strain_names,
            interaction_matrix_spec=interaction_matrix,
            introduction_age_masks=masks,
            strain_specs={_name_of(strain): strain for strain in strains},
        )

    @property
    def n_strains(self) -> int:
        return len(self.strain_names)

    @property
    def has_introductions(self) -> bool:
        return bool(self.introduced_strain_names)

    def has_strain(self, name: str) -> bool:
        return name in self.strains_to_idx

    def index_of(self, strain_name: str) -> int:
        try:
            return self.strains_to_idx[strain_name]
        except KeyError as exc:
            raise KeyError(
                f"Unknown strain {strain_name!r}. Known strains are: {self.strain_names}."
            ) from exc

    def get_strain_spec(self, strain_name: str) -> Any:
        try:
            return self.strain_specs[strain_name]
        except KeyError as exc:
            raise KeyError(
                f"Unknown strain {strain_name!r}. Known strains are: {self.strain_names}."
            ) from exc

    def interaction_spec(
        self,
        source_strain: str,
        target_strain: str,
    ) -> Any:
        source_idx = self.index_of(source_strain)
        target_idx = self.index_of(target_strain)

        return self.interaction_matrix_spec[source_idx][target_idx]

    def interaction_matrix_named(self) -> dict[str, dict[str, Any]]:
        return {
            source_name: {
                target_name: self.interaction_spec(source_name, target_name)
                for target_name in self.strain_names
            }
            for source_name in self.strain_names
        }

    def evaluate_interaction_matrix(
        self,
        context: Mapping[str, Any] | None = None,
        data: Any | None = None,
    ) -> Any:
        rows: list[list[Any]] = []

        for row in self.interaction_matrix_spec:
            evaluated_row: list[Any] = []

            for interaction in row:
                evaluate = getattr(interaction, "evaluate", None)

                if callable(evaluate):
                    evaluated_row.append(
                        evaluate(
                            context=context,
                            data=data,
                        )
                    )
                else:
                    evaluated_row.append(interaction)

            rows.append(evaluated_row)

        return jnp.asarray(rows)

    def introduction_age_mask(self, strain_name: str) -> tuple[int, ...]:
        self.index_of(strain_name)
        return self.introduction_age_masks[strain_name]

    def introduction_age_mask_matrix(self, dtype: Any = jnp.int32) -> Any:
        if self.n_age_bins_for_introduction == 0:
            return jnp.zeros((self.n_strains, 0), dtype=dtype)

        return jnp.asarray(
            [
                self.introduction_age_masks[strain_name]
                for strain_name in self.strain_names
            ],
            dtype=dtype,
        )


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
            _readonly_mapping(self.metadata),
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
            spec.parameters.transmission,
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
        return self.spec.parameters.solver

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


__all__ = [
    "RuntimeDimension",
    "RuntimeCompartment",
    "StateLayout",
    "RuntimeParameterLayout",
    "RuntimeTransmission",
    "RuntimeModel",
]
