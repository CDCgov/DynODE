import jax.numpy as jnp
import pytest
from scipy.optimize import root_scalar

from dynode.simulation import simulate
from examples.sir import get_config, get_odeparams, sir_ode


@pytest.mark.parametrize(
    "s0,i0,r0",
    [
        (0.99, 0.01, 0.0),
        (0.95, 0.05, 0.0),
        (0.90, 0.10, 0.0),
        (0.80, 0.20, 0.0),
    ],
)
def test_final_epidemic_size_matches_theory(s0, i0, r0):
    """
    Test that the final epidemic size matches the theoretical value from SIR model literature.
    For a basic SIR model, the final size equation is:
        R_inf = 1 - S_inf
        S_inf = S0 * exp(-R0 * (1 - S_inf))
    We numerically solve for S_inf and compare to simulation.
    """
    config = get_config()
    ode_params = get_odeparams(config)
    initial_state = config.initializer.get_initial_state(
        s_0=s0, i_0=i0, r_0=r0
    )
    solver_params = config.parameters.solver_params

    sol = simulate(
        ode=sir_ode,
        duration_days=300,
        initial_state=initial_state,
        ode_parameters=ode_params,
        solver_parameters=solver_params,
    )
    # sol.ys is a tuple of arrays (s, i, r), each shape (timesteps, 1)
    _, _, r = [
        arr.squeeze() for arr in sol.ys
    ]  # s, i are not needed for this test
    S0 = s0
    R0_param = config.parameters.transmission_params.strains[0].r0

    # Theoretical final susceptible fraction S_inf solves: S_inf = S0 * exp(-R0 * (1 - S_inf))
    # We'll solve this numerically
    def final_size_eq(S_inf):
        return S_inf - S0 * jnp.exp(-R0_param * (1 - S_inf))

    sol = root_scalar(
        lambda x: float(final_size_eq(x)),
        bracket=[0.0, S0],
        method="bisect",
        xtol=1e-8,
    )
    S_inf_theory = sol.root
    R_inf_theory = 1 - S_inf_theory

    # Simulated final recovered fraction
    R_inf_sim = float(r[-1])

    # Allow a small tolerance due to numerical integration
    assert pytest.approx(R_inf_sim, abs=2e-2) == R_inf_theory


@pytest.mark.parametrize(
    "s0,i0,r0",
    [
        (0.99, 0.01, 0.0),
        (0.95, 0.05, 0.0),
        (0.90, 0.10, 0.0),
        (0.80, 0.20, 0.0),
        (0.8, 0.0, 0.2),  # no infections, no epidemic
        (0.75, 0.1, 0.15),
    ],
)
def test_sir_mass_conservation(s0, i0, r0):
    """
    Regression test: The sum S+I+R should remain constant (within tolerance) over time.
    """
    config = get_config()
    ode_params = get_odeparams(config)
    initial_state = config.initializer.get_initial_state(
        s_0=s0, i_0=i0, r_0=r0
    )
    solver_params = config.parameters.solver_params

    sol = simulate(
        ode=sir_ode,
        duration_days=120,
        initial_state=initial_state,
        ode_parameters=ode_params,
        solver_parameters=solver_params,
    )
    s, i, r = [arr.squeeze() for arr in sol.ys]
    total = s + i + r
    # Should be constant and equal to initial sum
    assert jnp.allclose(total, total[0], atol=1e-6)
