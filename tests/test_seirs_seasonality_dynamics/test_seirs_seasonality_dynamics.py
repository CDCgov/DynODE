import jax.numpy as jnp
import pytest

from dynode import simulate
from examples.seirs_seasonal_forcing import (
    get_config,
    get_seirs_odeparams,
    seirs_ode_seasonal,
)


@pytest.mark.parametrize(
    "r_0, infectious_period, latent_period, waning_period",
    [
        (2.0, 7.0, 3.0, 60.0),
        (3.0, 5.0, 2.0, 100.0),
    ],
)
def test_seirs_endemic_equilibrium(
    r_0, infectious_period, latent_period, waning_period
):
    config = get_config(
        r_0=r_0,
        infectious_period=infectious_period,
        latent_period=latent_period,
        waning_period=waning_period,
    )
    # run simulation long enough to ensure equibilbrium
    ode_params = get_seirs_odeparams(config)
    sol = simulate(
        ode=seirs_ode_seasonal,
        duration_days=1000,
        initial_state=config.initializer.get_initial_state(),
        ode_parameters=ode_params,
        solver_parameters=config.parameters.solver_params,
    )
    s, e, i, r = [arr.squeeze() for arr in sol.ys]
    # Take the last 100 days and ensure that we are still modulating with seasonality
    assert jnp.std(s[-100:]) > 1e-4
    assert jnp.std(e[-100:]) > 1e-4
    assert jnp.std(i[-100:]) > 1e-4
    assert jnp.std(r[-100:]) > 1e-4
