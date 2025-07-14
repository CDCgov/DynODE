# DynODE Configuration Module

The DynODE configuration module defines the structure and validation logic for compartmental ODE models. It provides a flexible, type-safe way to specify model compartments, their dimensions, and the bins (states) within each dimension. This documentation covers the core classes: `SimulationConfig`, `Compartment`, `Dimension`, and `Bin`.

Lets start with a visual representation of the configuration structure, then go into each class in detail.

---

## Visual Representation

```{mermaid}
%%{ init: { "theme": "dark", "themeVariables": { "primaryColor": "#bb86fc", "background": "#121212"} } }%%
classDiagram
    class SimulationConfig {
        +list~Compartment~ compartments
        +Params parameters
        +Initializer initializer
        +NameSpace idx
        +get_compartment(name)
        +flatten_bins() list~Bin~
        +flatten_dims() list~Dimension~
    }
    class Compartment {
        +str name
        +list~Dimension~ dimensions
        +tuple~int~ shape
        +NameSpace idx
    }
    class Dimension {
        +str name
        +list~Bin~ bins
        +__len__()
        +idx
    }
    class Bin {
        +str name
    }
    class Params {
        +SolverParams solver_params
        +TransmissionParams transmission_params
    }
    class SolverParams {
        +Object solver_method
        +float ode_solver_rel_tolerance
        +float ode_solver_abs_tolerance
        +int max_steps
        +float constant_step_size
        +list~float~ discontinuity_points
    }
    class TransmissionParams {
        +list~Strain~ strains
        +dict strain_interactions
        +contact_matrix
    }
    class Strain {
        +str strain_name
        +r0
        +infectious_period
    }
    class Initializer {
        +str description
        +float population_size
        +date initialize_date
        +get_initial_state() CompartmentState
    }

    SimulationConfig --> Compartment
    SimulationConfig --> Params
    SimulationConfig --> Initializer
    Compartment --> Dimension
    Dimension --> Bin
    Params --> SolverParams
    Params --> TransmissionParams
    TransmissionParams --> Strain
```
---

## Bin

A `Bin` represents a single discrete state within a dimension. This could be an age group, vaccination status, or any other stratifying factor.

**Key Fields:**
- `name`: Unique name for the bin within its dimension.

**Specialized Bins:**
- `DiscretizedPositiveIntBin`: Bin with inclusive integer min/max values (e.g., for age or dose count).
- `AgeBin`: Specialized `DiscretizedPositiveIntBin` for age groups, with auto-generated names.
- `WaneBin`: Bin for waning immunity, with fields for waiting time and base protection.

**Validation:**
- For discretized bins, ensures `min_value <= max_value`.

---

## Dimension

A `Dimension` defines an axis of stratification within a compartment (e.g., age, vaccination, immune history).

**Key Fields:**
- `name`: Unique name for the dimension within a compartment.
- `bins`: List of `Bin` objects representing the discrete states along this dimension.

**Key Methods:**
- `__len__`: Number of bins in the dimension.
- `.idx`: Property returning a namespace mapping bin names to their indices within the `bins` list.

**Validation:**
- All bins must be of the same type and have unique names.
- For `DiscretizedPositiveIntBin`-based dimensions, bins must be sorted in increasing order, non-overlapping, and gapless.

**Specialized Dimensions:**
- `VaccinationDimension`: For ordinal and seasonal vaccination tracking.
- `ImmuneHistoryDimension`, `FullStratifiedImmuneHistoryDimension`, `LastStrainImmuneHistoryDimension`: For tracking infection history.
- `WaneDimension`: For tracking waning immunity with custom waiting times and protection levels.

---

## Compartment

A `Compartment` represents a single population group or state in the model, defined by a set of stratifying dimensions (e.g., age, vaccination status).

**Key Fields:**
- `name`: Unique name for the compartment.
- `dimensions`: List of `Dimension` objects defining the axes of the compartment.

**Key Methods:**
- `.shape`: Returns the shape of the compartment (tuple of bin counts per dimension).
- `.idx`: Cached property for enum-like access to dimension and bin indices.

**Validation:**
- Ensures all dimension names are unique within the compartment.

---

## Initializer
An `Initializer` defines how the initial state of the model is set up. It can be a simple constant value, or more complex initialization logic. The logic itself is defined mostly in the `get_initial_state()` method, which must be implemented by subclasses.

**Key Fields:**
- `description`: A human-readable description of the initializer, its data sources, and intended initialization date range.
- `initialize_date`: The date at which the model should be initialized.
- `population_size`: The total population size to distribute across compartments at initialization.

**Key Methods:**
- `get_initial_state(**kwargs)`: Abstract method to generate the initial compartment state, ensuring values sum to `population_size`. Must be implemented by subclasses.

**Validation:**
- Ensures `population_size` is a positive integer.
- Enforces implementation of `get_initial_state()` in subclasses.

---

## Params: SolverParams

`SolverParams` specifies the configuration for the ODE solver used in simulations.

**Key Fields:**
- `solver_method`: The ODE solver algorithm (e.g., `Tsit5`). Defaults to a general-purpose, non-stiff solver.
- `ode_solver_rel_tolerance` / `ode_solver_abs_tolerance`: Relative and absolute tolerances for adaptive step size control.
- `max_steps`: Maximum number of steps the solver will take before raising an error.
- `constant_step_size`: If nonzero, uses a fixed step size; otherwise, adaptive stepping is used.
- `discontinuity_points`: List of time points where the system has known discontinuities (e.g., intervention changes).

**Validation:**
- Ensures all solver parameters are positive and consistent with solver requirements.

---

## Params: TransmissionParams

`TransmissionParams` defines the structure for transmission-related parameters, especially for models with multiple pathogen strains.

**Key Fields:**
- `strain_interactions`: Nested dictionary specifying interaction parameters between strains (e.g., cross-immunity, competition). Keys are strain names; values are dictionaries mapping other strain names to interaction values or distributions.
- `strains`: List of `Strain` objects, each representing a pathogen variant or lineage.

**Validation:**
- Ensures the `strains` list is not empty.
- Validates that `strain_interactions` covers all strains and is symmetric (all strains interact with all others, including themselves).
- Checks that certain optional fields (e.g., `exposed_to_infectious`, `vaccine_efficacy`) are either set for all strains or none, and that introduction ages are consistent across strains.

---

## Params

The `Params` class is a container for all model parameters, grouping together `SolverParams` and `TransmissionParams`.

---

## SimulationConfig

`SimulationConfig` is the top-level configuration object for a DynODE model. It encapsulates the entire model structure, including compartments, parameters, and initialization logic.

**Key Fields:**
- `initializer`: An `Initializer` object specifying how the initial state is created.
- `compartments`: A list of `Compartment` objects, each representing a population or state group. Think of these objects as a wireframe for the compartments in the model, but are not the values themselves.
- `parameters`: A `Params` object containing model parameters.

**Key Methods:**
- `get_compartment(name)`: Retrieve a compartment by name.
- `flatten_bins()`: Returns a flat list of all bins in all compartments.
- `flatten_dims()`: Returns a flat list of all dimensions in all compartments.

**Validation:**
- Ensures unique compartment names.
- Ensures dimensions with the same name across compartments are structurally identical.
- Validates immune history dimensions and strain introduction logic.

**Indexing:**
- `.idx` property provides a cached, enum-like namespace for programmatic access to compartments and their dimensions/bins. This enum is recursive, meaning `config.idx.s` will return the index of the `s` compartment within `config.compartments`, but you can also access `config.idx.s.age` to get the index of the `age` dimension within the `s` compartment, and furthermore `config.idx.s.age.under_5` to get the index of the `under_5` bin within the `age` dimension of the `s` compartment.

---


## Example

lets define a simple SIR model with a single strain and a simple initializer. This example is repeated in the `examples/sir.py` file, but is included here for completeness.

```python
from dynode.config import (
    SimulationConfig,
    Compartment,
    Dimension,
    AgeBin,
    Params,
    SolverParams,
    TransmissionParams,
    Strain,
    Initializer,
)
import jax.numpy as jnp
from datetime import date

# --- SIR Initializer with age stratification ---
class SIRInitializer(Initializer):
    """Initializer for SIR model, setting initial conditions for compartments."""

    def __init__(self):
        """Create an SIR Initializer."""
        super().__init__(
            description="An SIR initalizer",
            initialize_date=date(2022, 2, 11),  # random date
            population_size=1000,
        )

    def get_initial_state(
        self, s0_prop=0.99, i0_prop=0.01, **kwargs
    ) -> CompartmentState:
        """Get initial compartment values for an SIR model stratified by age."""
        assert s0_prop + i0_prop == 1.0, (
            "s0_prop and i0_prop must sum to 1.0, "
            f"got {s0_prop} and {i0_prop}."
        )
        # proportion of young to old in the population
        age_demographics = jnp.array([0.75, 0.25])
        num_susceptibles = self.population_size * jnp.array([s0_prop])
        s_0 = num_susceptibles * age_demographics
        num_infectious = self.population_size * jnp.array([i0_prop])
        i_0 = num_infectious * age_demographics
        r_0 = jnp.array([0.0, 0.0])
        # SimulationConfig has no impact on initial state in this example
        return (s_0, i_0, r_0)


# --- SIRConfig for bin definitions and strain specification---
dimension = Dimension(
    name="age", bins=[Bin(name="young"), Bin(name="old")]
)
s = Compartment(name="s", dimensions=[dimension])
i = Compartment(name="i", dimensions=[dimension])
r = Compartment(name="r", dimensions=[dimension])
strain = [
    Strain(strain_name="swo9", r0=r_0, infectious_period=infectious_period)
]
contact_matrix = jnp.array([[0.7, 0.3], [0.3, 0.7]])
parameters = Params(
    solver_params=SolverParams(),
    transmission_params=TransmissionParams(
        strains=strain,
        strain_interactions={"swo9": {"swo9": 1.0}},
        contact_matrix=contact_matrix,
    ),
)
config = SimulationConfig(
    compartments=[s, i, r],
    initializer=SIRInitializer(),
    parameters=parameters,
)
```

---

For further details, see the docstrings in each class and the validation logic in the source files.
