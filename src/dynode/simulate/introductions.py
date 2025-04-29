import jax.numpy as jnp
from jax import Array
from jax.scipy.stats.norm import pdf
from jax.typing import ArrayLike


def external_i(
    t: ArrayLike,
    introduction_times: Array,
    introduction_scales: Array,
    introduction_percentages: Array,
    introduction_ages: Array,
    population_by_age: Array,
    i_shape: tuple,
) -> Array:
    """Introduce strains via an interacting external population.

    Parameters
    ----------
    t : float
        current simulation day

    introduction_times : Array
        introduction time as a float for each strain in the model. Negative
        introduction times are allowed and if sufficiently large corresponding
        strains will not be introduced. Shape=(num_strains,)

    introduction_scales : Array
        Standard deviation of introduction curve, in days, for each strain.
        Shape=(num_strains,)

    introduction_percentages : Array
        magnitude of externally infected individuals introduced over the entire
        introduction curve, for that particular strain. Magnitude is determined
        by multiplying this percentage by the population in `population_by_age`.
        Shape=(num_strains,)

    introduction_ages : Array
        vector of binary age group masks to dictate the age structure of
        externally introduced population. Shape=(num_strains,num_age_groups)

    population_by_age : Array
        population counts of each age group.Shape=(num_age_groups,)

    i_shape : tuple
        shape of the i compartment to match.

    Returns
    -------
    Array
    Array matching shape of `i_shape` containing externally introduced
    population for each strain for time `t`.

    """
    ext_i = jnp.zeros(i_shape)
    densities = pdf(t, loc=introduction_times, scale=introduction_scales)
    introduction_densities = introduction_percentages * densities
    populations = introduction_ages * population_by_age
    populations_new_infections = introduction_densities[:, None] * populations
    ext_i = ext_i.at[:, 0, 0, :].set(populations_new_infections.T)
    return ext_i
