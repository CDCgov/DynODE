"""Top level classes for DynODE configs."""

from functools import cached_property
from types import SimpleNamespace
from typing import List, Union

import jax.numpy as jnp
from jax import Array
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)
from typing_extensions import Any, Self

from dynode.typing import DynodeName

from .bins import AgeBin, Bin
from .dimension import (
    Dimension,
    FullStratifiedImmuneHistoryDimension,
    ImmuneHistoryDimension,
    LastStrainImmuneHistoryDimension,
)
from .initializer import Initializer
from .params import Params
from .simulation_date import (
    replace_simulation_dates,
    set_dynode_init_date_flag,
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
        assert len(set(dimension_names)) == len(dimension_names), (
            "you can not have two identically named dimensions within a compartment"
        )
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


class SimulationConfig(BaseModel):
    """An ODE compartment model configuration file."""

    # allow users to pass custom types into SimulationConfig
    model_config = ConfigDict(arbitrary_types_allowed=True)
    initializer: Initializer = Field(
        description="""Initializer to create initial state with."""
    )
    compartments: List[Compartment] = Field(
        description="""Compartments of the model."""
    )
    parameters: Params = Field(
        description="""Model parameters, includes epidemiological and miscellaneous."""
    )

    def model_post_init(self, __context: Any) -> None:
        """Post-initialization method to replace instances of SimulationDate with numeric sim days."""
        init_date = self.initializer.initialize_date
        set_dynode_init_date_flag(init_date)
        self = replace_simulation_dates(self)

    @cached_property
    def idx(self):
        """An enum-like structure for compartments and their dimensions.

        Note
        ----
        This is a cache property, so it will only be computed once, modifications
        to the compartments will not change the enum after it is created.

        Returns
        -------
            SimpleNamespace: A namespace containing compartments and their dimensions.
        """
        compartments_namespace = SimpleNamespace()
        for compartment_idx, compartment in enumerate(self.compartments):
            # build up the bins namespace for this compartment
            compartment_obj = _IntWithAttributes(
                compartment_idx, **compartment.idx.__dict__
            )
            setattr(compartments_namespace, compartment.name, compartment_obj)
        return compartments_namespace

    @model_validator(mode="after")
    def _validate_no_shared_compartment_names(self) -> Self:
        """Validate that no two compartments have the same name."""
        compartment_names = [c.name for c in self.compartments]
        assert len(set(compartment_names)) == len(compartment_names), (
            f"you can not have two identically named compartments, "
            f"found shared names: "
            f"{set([x for x in compartment_names if compartment_names.count(x) > 1])}"
        )
        return self

    @model_validator(mode="after")
    def _validate_shared_compartment_dimensions(self) -> Self:
        """Validate that any dimensions with same name across compartments are equal."""
        # quad-nested for loops are not ideal, but lists are very small so this should be fine
        dimension_map: dict[str, Dimension] = {}
        all_dims: list[Dimension] = self.flatten_dims()
        for dimension in all_dims:
            if dimension.name in dimension_map:
                assert (
                    dimension == dimension_map[dimension.name]
                ), f"""dimension {dimension.name} has different definitions
                across different compartments, if this intended, make
                the dimensions have different names"""
            else:  # first time encountering this dimension name
                dimension_map[dimension.name] = dimension
        return self

    @model_validator(mode="after")
    def _validate_immune_histories(self) -> Self:
        """Validate that the immune history dimensions within each compartment are initialized from the same strain definitions.

        Example
        -------
        If you have 2 strains, `x` and `y`,
        - a `FullStratifiedImmuneHistory` should have 4 bins, `none`, `x`, `y`, `x_y`
        - a `LastStrainImmuneHistory` should have 3 bins, `none`, `x`, `y`
        - Neither class should bins with any other strain `z` or exclude one of the required bins.
        """
        strains = self.parameters.transmission_params.strains
        # gather all ImmuneHistory dimensions
        all_dims = self.flatten_dims()
        all_immune_hist_dims = [
            d for d in all_dims if isinstance(d, ImmuneHistoryDimension)
        ]
        # assert that all immune histories were generated from this set of strains.
        for dimension in all_immune_hist_dims:
            assert isinstance(
                dimension,
                (
                    FullStratifiedImmuneHistoryDimension,
                    LastStrainImmuneHistoryDimension,
                ),
            )
            assert type(dimension)(strains) == dimension, (
                "Found immune states that dont correlate with strains from transmission_params"
            )
        return self

    @model_validator(mode="after")
    def _create_introduction_ages_mask_encoding(self) -> Self:
        """Parse Strain's introduction_ages to a binary mask."""
        # dont bother one-hot encoding introduction_ages if they dont exist
        if any(
            [
                strain.introduction_ages is not None
                for strain in self.parameters.transmission_params.strains
            ]
        ):
            # find a dimension with Age stratification
            age_binning = []
            for dim in self.flatten_dims():
                # only check first element since dimensions must all be same type
                if isinstance(dim.bins[0], AgeBin):
                    age_binning = dim.bins
                    break
            assert len(age_binning) > 0, (
                "attempted to encode introduction_ages but could not "
                "find any age structure in the compartments"
            )
            mask = []
            for strain in self.parameters.transmission_params.strains:
                # assume intro_ages is found in age_structure due to above validator
                if strain.introduction_ages is not None:
                    mask = [
                        1 if b in strain.introduction_ages else 0
                        for b in age_binning
                    ]
                else:
                    mask = [0 for _ in age_binning]
                # set the private field now that validation is complete.
                strain.introduction_ages_mask_vector = mask
        return self

    @model_validator(mode="after")
    def _validate_introduced_strains(self) -> Self:
        """Validate that all introduced strains have the same age binning as defined by the Model's compartments."""
        strains = self.parameters.transmission_params.strains
        all_bins = self.flatten_bins()
        age_structure = [b for b in all_bins if isinstance(b, AgeBin)]
        for strain in strains:
            strain_target_ages = strain.introduction_ages
            if strain.is_introduced and strain_target_ages is not None:
                assert all(
                    [
                        target_age in age_structure
                        for target_age in strain_target_ages
                    ]
                ), (
                    f"{strain.strain_name} attempts to introduce itself using "
                    f"{strain_target_ages} age bins, but those are not found "
                    "within the age structure of the model."
                )
        return self

    def get_compartment(self, compartment_name: str) -> Compartment:
        """Search for and return a Compartment if it exists.

        Parameters
        ----------
        compartment_name : str
            name of the compartment to return

        Returns
        -------
        Compartment
            Compartment class with matching name.

        Raises
        ------
        AssertionError
            raise if `compartment_name` not found within `self.compartments`
        """
        for compartment in self.compartments:
            if compartment_name == compartment.name:
                return compartment
        raise AssertionError(
            "Compartment with name %s not found in model, found only these names: %s"
            % (compartment_name, str([c.name for c in self.compartments]))
        )

    def flatten_bins(
        self,
    ) -> Union[list[Bin]]:
        """Flatten all compartments down to list of bins.

        Returns
        -------
        list[Bin]
            flattened compartments' bin objects.

        Note
        ----
        This operation preserves the order of the compartments, dimensions,
        and bins in the final flattened output.
        """
        flattened_lst: list[Bin] = []
        for compartment in self.compartments:
            # flatten bins for each dimension
            for dimension in compartment.dimensions:
                flattened_lst.extend(dimension.bins)
        return flattened_lst

    def flatten_dims(
        self,
    ) -> Union[list[Dimension]]:
        """Flatten all compartments down to list of dimensions.

        Returns
        -------
        list[Dimension]
            flattened compartments' Dimension objects.

        Note
        ----
        This operation preserves the order of the compartments,
        and dimensions, in the final flattened output.
        """
        flattened_lst: list[Dimension] = []
        for compartment in self.compartments:
            flattened_lst.extend(compartment.dimensions)
        return flattened_lst
