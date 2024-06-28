# needs to exist to define a module
import jax

SEIC_Compartments = tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]
