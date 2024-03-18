from functools import partial

import jax
import jax.numpy as jnp

import utils


def make_1strain_init_state(ori_init_state):
    """
    Take original (multi-strain) initial state, and squash them down to
    one-strain initial state. This function is higly specific to this
    experiment.
    """
    nstrain = 3
    new_init_state = []
    for st in ori_init_state:
        shp = list(st.shape)
        shp[1] = 2
        shp[-1] = 1 if shp[-1] == nstrain else shp[-1]
        shp = tuple(shp)
        newst = jnp.zeros(shp)
        if shp[-1] == 1:
            st_1strain = jnp.sum(st, axis=-1)
            newst = newst.at[:, 0, :, 0].set(st_1strain[:, 0, :])
            newst = newst.at[:, 1, :, 0].set(
                jnp.sum(st_1strain[:, 1:, :], axis=1)
            )
        else:
            newst = newst.at[:, 0, :, :].set(st[:, 0, :, :])
            newst = newst.at[:, 1, :, :].set(jnp.sum(st[:, 1:, :, :], axis=1))
        new_init_state = new_init_state + [newst]

    return tuple(new_init_state)


@jax.jit
def deBoor(ind, x, k, c):
    """
    Evaluates S(x) = Value of cubic spline at x, using de Boor algorithm
    (https://en.wikipedia.org/wiki/De_Boor%27s_algorithm).

    Arguments
    ---------
    ind: Index of knot interval that contains x.
    x: Position.
    k: Array of knot positions, needs to be padded, i.e., adding 3 more
    boundary knots before and after an array of knots inclusive of
    boundary knots. E.g. [0, 0, 0, 0, 1, 2, 3, 3, 3, 3] with 0 and 3
    being the bounds.
    c: Array of control points (or regression coefficients) that is the
    length of the number of knots (inclusive of boundary knots) + 2.
    """
    d = [c[j + ind - 3] for j in range(0, 3 + 1)]

    for r in range(1, 3 + 1):
        for j in range(3, r - 1, -1):
            alpha = (x - k[j + ind - 3]) / (
                k[j + 1 + ind - r] - k[j + ind - 3]
            )
            d[j] = (1.0 - alpha) * d[j - 1] + alpha * d[j]

    return d[3]


@partial(jax.jit, static_argnums=(0))
def custom_beta_coef(self, t):
    """
    Evaluates the value of the cubic spline (formulated using B-spline)
    at time t. This function is specifically designed to override the
    existing beta_coef() function within the inferer class.
    """
    knots = self.config.BSPLINE_KNOTS
    coeffs = self.config.BSPLINE_COEFFS

    value = deBoor(jnp.searchsorted(knots, t, "right") - 1, t, knots, coeffs)
    return value


@partial(jax.jit, static_argnums=(0))
def vaccination_rate(self, t):
    """Returns a coefficient for the beta value for cases of seasonal forcing or external impacts
    onto beta not directly measured in the model. e.g., masking mandates or holidays.
    Currently implemented via an array search with timings BETA_TIMES and coefficients BETA_COEFICIENTS

    Parameters
    ----------
    t: float as Traced<ShapedArray(float32[])>
        current time in the model. Due to the just-in-time nature of Jax this float value may be contained within a
        traced array of shape () and size 1. Thus no explicit comparison should be done on "t".

    Returns:
    coefficient by which BETA can be multiplied to externally increase or decrease the value to account for measures or seasonal forcing.
    """
    # a smart lookup function that works with JAX just in time compilation
    # if t > self.config.BETA_TIMES_i, return self.config.BETA_COEFICIENTS_i
    t_added = getattr(self.config, "VAX_MODEL_DAYS_SHIFT", 0)
    t_mod = jnp.where(t + t_added < 0, 0, t + t_added)
    multiplier = jnp.where(t + t_added < 0, 0, 1)

    return (
        jnp.exp(
            utils.evaluate_cubic_spline(
                t_mod,
                self.config.VAX_MODEL_KNOT_LOCATIONS,
                self.config.VAX_MODEL_BASE_EQUATIONS,
                self.config.VAX_MODEL_KNOTS,
            )
        )
        * multiplier
    )
