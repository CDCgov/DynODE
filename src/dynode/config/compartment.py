from functools import cached_property
from types import SimpleNamespace
from typing import List

from jax import Array
from jax import numpy as jnp
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)
from typing_extensions import Self

from ..typing import DynodeName
from .dimension import (
    Dimension,
)


class Compartment(BaseModel):
    """Defines a single compartment of an ODE model."""

    # allow jax array objects within Compartments
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: DynodeName = Field(
        description="""Compartment name, must be unique within a CompartmentModel."""
    )
    dimensions: List[Dimension] = Field(
        description="""Compartment dimension definitions."""
    )
    values: Array = Field(
        default_factory=lambda: jnp.array([]),
        description="Compartment matrix values.",
    )

    @model_validator(mode="after")
    def _shape_match(self) -> Self:
        """Set default values if unspecified, asserts dimensions and values shape matches."""
        target_values_shape: tuple[int, ...] = tuple(
            [len(d_i) for d_i in self.dimensions]
        )
        if bool(self.values.any()):
            assert target_values_shape == self.values.shape
        else:
            # fill with default for now, values filled in at runtime.
            self.values = jnp.zeros(target_values_shape)
        return self

    @model_validator(mode="after")
    def _validate_dimensions_names(self):
        """Assert that all dimensions in the Compartment have unique names."""
        dimension_names = [dim.name for dim in self.dimensions]
        assert (
            len(set(dimension_names)) == len(dimension_names)
        ), "you can not have two identically named dimensions within a compartment"
        return self

    @property
    def shape(self) -> tuple[int, ...]:
        """Get shape of the compartment."""
        return tuple([len(d_i) for d_i in self.dimensions])

    @cached_property
    def idx(self):
        """An enum-like structure for dimensions and their bins.

        Note
        ----
        This is a cache property, so it will only be computed once, modifications
        to the compartment will not change the idx after it is created.

        Returns
        -------
            SimpleNamespace: A namespace containing dimensions and their bins.
        """
        dims_namespace = SimpleNamespace()
        for dim_idx, dimension in enumerate(self.dimensions):
            # save the dimension index along with the indexes of all the bins.
            dim_obj = _IntWithAttributes(dim_idx, **dimension.idx.__dict__)
            setattr(dims_namespace, dimension.name, dim_obj)
        return dims_namespace

    def __eq__(self, value) -> bool:
        """Check for equality definitions between two Compartments.

        Parameters
        ----------
        value : Any
            Other value to compare, usually another Compartment

        Returns
        -------
        bool
            whether or not the two compartments are equal in name and dimension structure.

        Note
        ----
        does not check the values of the compartments, only their dimensionality and definition.
        """
        if isinstance(value, Compartment):
            if self.name == value.name and len(self.dimensions) == len(
                value.dimensions
            ):
                # check both compartments have same dimensions in same order
                for dim_l, dim_r in zip(self.dimensions, value.dimensions):
                    if dim_l != dim_r:
                        return False
                return True
        return False

    def __setitem__(self, index: int | slice | tuple, value: float) -> None:
        """Set Compartment value in a numpy-like way.

        Parameters
        ----------
        index : int | slice | tuple
            index or slice or tuple to index the Compartment's values.
        value : float
            float to set values[index] to.
        """
        self.values = self.values.at[index].set(value)

    def __getitem__(self, index: int | slice | tuple) -> Array:
        """Get the Compartment's values at some index.

        Parameters
        ----------
        index : int | slice | tuple
            index to look up.

        Returns
        -------
        Any
            value of the `self.values` tensor at that index.
        """
        return self.values.at[index].get()


class _IntWithAttributes(int):
    """A subclass of int that allows setting attributes."""

    def __new__(cls, value, **attributes):
        obj = super().__new__(cls, value)
        for key, val in attributes.items():
            setattr(obj, key, val)
        return obj

    def __str__(self):
        return str(self.__dict__)
