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
from typing_extensions import Any, Self

from dynode.utils import set_dynode_init_date_flag

from .bins import AgeBin, Bin
from .compartment import Compartment, _IntWithAttributes
from .dimension import (
    Dimension,
    FullStratifiedImmuneHistoryDimension,
    ImmuneHistoryDimension,
    LastStrainImmuneHistoryDimension,
)
from .initializer import Initializer
from .params import Params


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
        """Initialize context for model run."""
        init_date = self.initializer.initialize_date
        set_dynode_init_date_flag(init_date)

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
            assert (
                type(dimension)(strains) == dimension
            ), "Found immune states that dont correlate with strains from transmission_params"
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
