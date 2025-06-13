import jax.numpy as jnp
import pytest

from dynode.simulation import simulate
from examples.seirs import get_config, get_seirs_odeparams, seirs_ode


def compute_seirs_endemic_equilibrium(beta, gamma, sigma, omega):
    """
    Computes the endemic equilibrium for the SEIRS model.
    Returns (S*, E*, I*, R*) for a normalized population (N=1).
    """
    S_star = gamma / beta
    denom = 1 + (gamma / sigma) + (gamma / omega)
    I_star = (1 - S_star) / denom
    E_star = (gamma / sigma) * I_star
    R_star = (gamma / omega) * I_star
    return S_star, E_star, I_star, R_star


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
        ode=seirs_ode,
        duration_days=1000,
        initial_state=config.initializer.get_initial_state(),
        ode_parameters=ode_params,
        solver_parameters=config.parameters.solver_params,
    )
    s, e, i, r = [arr.squeeze() for arr in sol.ys]
    # Take the last 100 days and average to estimate equilibrium
    # Ensure the last 100 days are stable (not fluctuating)
    assert jnp.std(s[-100:]) < 1e-4
    assert jnp.std(e[-100:]) < 1e-4
    assert jnp.std(i[-100:]) < 1e-4
    assert jnp.std(r[-100:]) < 1e-4
    s_eq = float(s[-1])
    e_eq = float(e[-1])
    i_eq = float(i[-1])
    r_eq = float(r[-1])
    # Compute theoretical equilibrium
    S_star, E_star, I_star, R_star = compute_seirs_endemic_equilibrium(
        ode_params.beta, ode_params.gamma, ode_params.sigma, ode_params.omega
    )
    # Assert close to theoretical values
    assert pytest.approx(s_eq, rel=1e-2) == S_star
    assert pytest.approx(e_eq, rel=1e-2) == E_star
    assert pytest.approx(i_eq, rel=1e-2) == I_star
    assert pytest.approx(r_eq, rel=1e-2) == R_star
