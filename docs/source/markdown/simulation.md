# DynODE Simulation Module

The `simulation` module provides utilities for solving systems of ordinary differential equations (ODEs) using the [diffrax](https://docs.kidger.site/diffrax/) library. For more information on diffrax in DynODE check out the it's section in the [back-end libraries section](https://github.com/CDCgov/DynODE/wiki/library_backend#diffrax)


## `simulate` Function

```python
simulate(
    ode: ODE_Eqns,
    duration_days: int,
    initial_state: CompartmentState,
    ode_parameters: AbstractODEParams,
    solver_parameters: SolverParams,
    sub_save_indices: Optional[Tuple[int, ...]] = None,
    save_step: int = 1,
) -> Solution
```

The `simulate` function numerically solves a user-supplied ODE system over a given number of days, returning a `diffrax.Solution` object containing the compartment states at each saved time point.

our `simulate` function is a thin wrapper around the diffrax [diffeqsolve](https://docs.kidger.site/diffrax/api/diffeqsolve/) method so users dont need to learn the libraries syntax.

### Parameters

- `ode`: user supplied callable ODE function with signature as follows
  - `def ode(t: float, y: CompartmentState, params: AbstractODEParams) -> CompartmentGradients`
- `duration_days`: Number of days to simulate.
- `initial_state`: Tuple of JAX arrays representing the initial state of each compartment.
- `ode_parameters`: [Chex](#chex) parameters object for the ODE, passed through to `params` in the ode.
- `solver_parameters`: a `SolverParams` object containing solver configuration, such as step size and error tolerances.
- `sub_save_indices`: Optional tuple of indices specifying which compartments to save.
- `save_step`: Interval (in days) at which to save the solution.

### Returns

- `diffrax.Solution`: Contains the time points and compartment states for the simulation. [Solution documentation](https://docs.kidger.site/diffrax/api/solution/)

### Notes

- All compartment states must be JAX arrays.
- The ODE function parameter order matters must accept parameters in the order: `(t, y, params)`.
