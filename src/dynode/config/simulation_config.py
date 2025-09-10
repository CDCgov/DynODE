"""Top level classes for DynODE configs."""

from functools import cached_property
from types import SimpleNamespace
from typing import List, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)
from typing_extensions import Self

from dynode.typing import DynodeName

from .bins import AgeBin, Bin
from .dimension import (
    Dimension,
    FullStratifiedImmuneHistoryDimension,
    ImmuneHistoryDimension,
    LastStrainImmuneHistoryDimension,
)
from .initializer import Initializer
from .parameter_set import ParameterSet
from .params import SolverParams
from .sample import sample_then_resolve


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
    # parameters: Params = Field(
    #    description="""Model parameters, includes epidemiological and miscellaneous."""
    # )
    parameter_sets: dict[str, ParameterSet]

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
        # need to add validator to ensure there is a parameter set of type TransmissionParams also need to handle fetching that set differently
        strains = self.parameter_sets["transmission_params"].strains
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
                for strain in self.parameter_sets[
                    "transmission_params"
                ].strains
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
            for strain in self.parameter_sets["transmission_params"].strains:
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
        strains = self.parameter_sets["transmission_params"].strains
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

    def inject_parameters(
        self, injection_parameter_set: ParameterSet, set_keys: List[str] = None
    ) -> None:
        """
        Injects parameters from `injection_parameter_set` into selected parameter sets.

        Parameters
        ----------
        injection_parameter_set : ParameterSet
            The source of parameters to inject.

        set_keys : List[str], optional
            A list of keys in `self.parameter_sets` to inject into.
            If None, injects into all parameter sets.
        """
        for key, parameter_set in self.parameter_sets.items():
            #            if set_keys is not None and key not in set_keys:
            #                continue  # Skip keys not in the target list
            if isinstance(parameter_set, SolverParams):
                continue

            merged_fields = {
                **parameter_set.model_dump(),
                **injection_parameter_set.model_dump(),
            }

            # Reconstruct the model safely
            try:
                new_param_set = parameter_set.__class__.model_validate(
                    merged_fields
                )
            except TypeError as e:
                print(e)
                # Fallback if validation fails due to abstract types
                new_param_set = parameter_set.__class__.model_construct(
                    **merged_fields
                )
            self.parameter_sets[key] = new_param_set

    def sample_then_resolve_parameters(self, prefix: str) -> None:
        for key, parameter_set in self.parameter_sets.items():
            if isinstance(parameter_set, SolverParams):
                continue

            parameter_set = sample_then_resolve(
                parameter_set, _prefix=f"{prefix}_"
            )
            print(key)
            print(parameter_set)
            self.parameter_sets[key] = parameter_set
