from functools import partial


import jax
import jax.numpy as jnp


def make_1strain_init_state(ori_init_state):
    new_init_state = []
    for st in ori_init_state:
        shp = list(st.shape)
        shp[1] = 2
        shp[-1] = 1 if shp[-1] == 3 else shp[-1]
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
def deBoor(k, x, t, c):
    """Evaluates S(x).

    Arguments
    ---------
    k: Index of knot interval that contains x.
    x: Position.
    t: Array of knot positions, needs to be padded as described above.
    c: Array of control points.
    p: Degree of B-spline.
    """
    d = [c[j + k - 3] for j in range(0, 3 + 1)]

    for r in range(1, 3 + 1):
        for j in range(3, r - 1, -1):
            alpha = (x - t[j + k - 3]) / (t[j + 1 + k - r] - t[j + k - 3])
            d[j] = (1.0 - alpha) * d[j - 1] + alpha * d[j]

    return d[3]


@partial(jax.jit, static_argnums=(0))
def custom_beta_coef(self, t):
    knots = self.config.BSPLINE_KNOTS
    coeffs = self.config.BSPLINE_COEFFS

    value = deBoor(jnp.searchsorted(knots, t, "right") - 1, t, knots, coeffs)
    return value
