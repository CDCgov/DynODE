from datetime import date

from pydantic import BaseModel, Field, PositiveInt

from dynode.typing import CompartmentState


# TODO, do we really need this to be a class, and does the SimulationConfig need a reference to it
# how does this play when we are "sampling" initial state from a posterior particle in a previous fit
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

    def get_initial_state(self, **kwargs) -> CompartmentState:
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
