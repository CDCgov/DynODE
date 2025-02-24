"""Top level classes for DynODE configs."""

from datetime import date
from typing import Callable, List, Optional

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

from dynode import CompartmentGradiants

from .dimension import (
    Dimension,
    FullStratifiedImmuneHistory,
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
    def validate_shared_compartment_dimensions(self) -> Self:
        """Validate that any dimensions with same name across compartments are equal."""
        # quad-nested for loops are not ideal, but lists are very small so this should be fine
        dimension_map: dict[str, Dimension] = {}
        for compartment in self.compartments:
            for dimension in compartment.dimensions:
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
    def _validate_immune_histories(self):
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
        for compartment in self.compartments:
            for dimension in compartment.dimensions:
                dim_class = type(dimension)
                if (
                    dim_class is FullStratifiedImmuneHistory
                    or dim_class is LastStrainImmuneHistory
                ):
                    assert (
                        dim_class(strains) == dimension
                    ), "Found immune states that dont correlate with strains from transmission_params"
        return self

    @model_validator(mode="after")
    def _validate_vaccination_counts(self):
        """Validate that the number of doses you specify in your vaccination
        dimensions are consistent."""
        # assert that all similarly named dimensions have same vaccine bins
        num_shots = {}
        for compartment in self.compartments:
            for dimension in compartment.dimensions:
                dim_class = type(dimension)
                if dim_class is VaccinationDimension:
                    if dimension.name in num_shots:
                        assert (
                            dimension.max_shots == num_shots[dimension.name]
                        ), "vaccination dimensions with same name have different numbers of shots."
                    else:
                        num_shots[dimension.name] = dimension.max_shots

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
