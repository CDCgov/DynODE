# needs to exist to define a module
import jax

"""
SEIC Compartments defines a tuple of the four major compartments used in the model
S: Susceptible, E: exposed, I: Infectious, C: cumulative (book keeping)
the dimension definitions of each of these compartments is
defined by the following Enums within the global configuration file
S: S_AXIS_IDX
E/I/C: I_AXIS_IDX
The exact sizes of each of these dimensions also depend on the implementation and config file
"""
SEIC_Compartments = tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]
# a timeseries is a tuple of compartment sizes where the leading dimension is time
# so SEIC_Timeseries has shape (tf, SEIC_Compartments.shape) for some number of timesteps tf
SEIC_Timeseries = tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]
