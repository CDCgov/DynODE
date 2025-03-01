"""Top level classes for DynODE configs."""

from datetime import date
from typing import Callable, List, Optional, Union

from jax import Array
from jax import numpy as jnp
from numpyro.infer import MCMC, SVI
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveFloat,
    PositiveInt,
    field_validator,
    model_validator,
)
from typing_extensions import Self

from dynode.typing import CompartmentGradiants

from .bins import AgeBin, Bin
from .dimension import (
    Dimension,
    FullStratifiedImmuneHistory,
    ImmuneHistoryDimension,
    LastStrainImmuneHistory,
    VaccinationDimension,
)
from .params import InferenceParams, Params


class Compartment(BaseModel):
    """Defines a single compartment of an ODE model."""

    # allow jax array objects within Compartments
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = Field(
        description="""Compartment name, must be unique within a CompartmentModel."""
    )
    dimensions: List[Dimension] = Field(
        description="""Compartment dimension definitions."""
    )
    values: Array = Field(
        default_factory=lambda: jnp.array([]),
        description="Compartment matrix values.",
    )

    @field_validator("name", mode="before")
    @classmethod
    def _verify_names(cls, value: str) -> str:
        """Validate to ensure names are always lowercase and underscored."""
        if value.replace("_", "").isalpha():
            return value.lower()
        else:
            raise ValueError(
                "the name field must not contain non-alpha chars with the exception of underscores"
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

    @property
    def shape(self) -> tuple[int, ...]:
        """Get shape of the compartment."""
        return tuple([len(d_i) for d_i in self.dimensions])

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


class Initializer(BaseModel):
    """Initalize compartment state of an ODE model."""

    description: str = Field(
        description="""Description of the initializer, its data streams and/or
         its intended initialization date range."""
    )
    initialize_date: date = Field(description="""Initialization date.""")
    population_size: PositiveInt = Field(
        description="""Target initial population size."""
    )

    def get_initial_state(self, **kwargs) -> list[Compartment]:
        """Fill in compartments with values summing to `population_size`.

        Parameters
        ----------
        kwargs
            Any parameters needed by the specific initializer.

        Returns
        -------
        list[Compartment]
            input compartments with values filled in with compartments
            at `initialize_date`.

        Raises
        ------
        NotImplementedError
            Each initializer must implement their own `get_initial_state()`
            based on the available data streams on the `initialize_date`

        """
        raise NotImplementedError(
            "implement functionality to get initial state"
        )


class CompartmentalModel(BaseModel):
    """An ODE compartment model configuration file."""

    # allow users to pass custom types into CompartmentalModel
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
    # passed to diffrax.diffeqsolve
    ode_function: Callable[
        [List[Compartment], PositiveFloat, Params], CompartmentGradiants
    ] = Field(
        description="""Callable to calculate instantaneous rate of change of
        each compartment."""
    )

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
                (FullStratifiedImmuneHistory, LastStrainImmuneHistory),
            )
            assert (
                type(dimension)(strains) == dimension
            ), "Found immune states that dont correlate with strains from transmission_params"
        return self

    @model_validator(mode="after")
    def _create_introduction_ages_one_hot_encoding(self) -> Self:
        """Convert Strain's introduction_ages to a one-hot encoded tensor."""
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
            assert (
                len(age_binning) > 0
            ), """attempted to one hot encode introduction_ages but could not
                find any age structure in the model"""
            one_hot_vector = []
            for strain in self.parameters.transmission_params.strains:
                # assume intro_ages is found in age_structure due to above validator
                if strain.introduction_ages is not None:
                    one_hot_vector = [
                        1 if b in strain.introduction_ages else 0
                        for b in age_binning
                    ]
                else:
                    one_hot_vector = [0 for _ in age_binning]
                # set the private field now that validation is complete.
                strain.introduction_ages_one_hot = one_hot_vector
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
                ), f"""{strain.strain_name} attempts to introduce itself using
                    {strain_target_ages} age bins, but those are not found
                    within the age structure of the model."""
        return self

    @model_validator(mode="after")
    def _validate_vaccination_counts(self) -> Self:
        """Validate vaccination dose definitions are correct across the model."""
        # assert that all similarly named dimensions have same vaccine bins
        num_shots: dict[str, int] = {}
        all_dims = self.flatten_dims()
        all_vax_dims = [
            d for d in all_dims if isinstance(d, VaccinationDimension)
        ]
        for dimension in all_vax_dims:
            if dimension.name in num_shots:
                assert (
                    dimension.max_shots == num_shots[dimension.name]
                ), "vaccination dimensions with same name have different numbers of shots."
            else:
                num_shots[dimension.name] = dimension.max_shots
        return self

    def get_compartment(self, compartment_name: str) -> Compartment:
        """Search the CompartmentModel and return a specific Compartment if it exists.

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


class InferenceProcess(BaseModel):
    """Inference process for fitting a CompartmentalModel to data."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    model: CompartmentalModel = Field(
        description="CompartmentalModel on which inference is performed."
    )
    # includes observation method, specified at runtime.
    inference_method: Optional[MCMC | SVI] = Field(
        default=None,
        description="""Inference method to execute,
        currently only MCMC and SVI supported""",
    )
    inference_parameters: InferenceParams = Field(
        description="""inference related parameters, not to be confused with
        CompartmentalModel parameters for solving ODEs."""
    )
