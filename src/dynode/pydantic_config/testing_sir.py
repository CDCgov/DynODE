# %%
from datetime import date

import jax.numpy as jnp
from diffrax import (
    ODETerm,
    PIDController,
    SaveAt,
    Solution,
    Tsit5,
    diffeqsolve,
)
from jax import jit

import dynode.pydantic_config.bins as bins
import dynode.pydantic_config.dimension as dimensions
from dynode.pydantic_config.config_definition import (
    Compartment,
    CompartmentalModel,
    Initializer,
    ParamStore,
)
from dynode.pydantic_config.strains import Strain


@jit
def sir_ode(state: list[Compartment], _, parameters: ParamStore):
    """Calculate instantanious rate of change of compartments within `state`.
    Parameters
    ----------
    state : PyTree
        PyTree of compartments representing current state of the model.
    _ : None
        Usually time at which calculating derivative, but this model does not
        use time as there are no time varying parameters.
    parameters : tuple[float]
        parameters to calculate flows with
    Returns
    -------
    Pytree
        PyTree of shape `state` containing flows in and out of each compartment
        in `state`.
    """
    # Unpack state
    s, i, r = [compartment.values for compartment in state]
    beta = parameters.strains[0].r0 / parameters.strains[0].infectious_period
    gamma = 1 / parameters.strains[0].infectious_period
    population = s + i + r

    # Compute flows
    ds_to_i = beta * s * i / population
    di_to_r = gamma * i

    # Compute derivatives
    ds = -ds_to_i
    di = ds_to_i - di_to_r
    dr = di_to_r

    return [ds, di, dr]


def solve_odes(
    initial_state: list[Compartment],
    args: ParamStore,
    tf: float | int,
    model,
) -> Solution:
    """Designates a Step Sizer and solves ODEs for `tf` days using `args` and `initial_state`.
    Parameters
    ----------
    initial_state : tuple[jax.Array]
        initial state of ODEs at t=0
    args : dict[str, any]
        _description_
    tf : float
        number of days to run for
    Returns
    -------
    diffrax.Solution
        Solution object of timeseries for each compartment.
    """
    # Solve ODE
    term = ODETerm(lambda t, state, parameters: model(state, t, parameters))
    solver = Tsit5()
    t0 = 0.0
    t1 = tf
    dt0 = 0.25
    times = jnp.linspace(t0, t1, int(tf) + 1)
    saveat = SaveAt(ts=times)
    solution = diffeqsolve(
        term,
        solver,
        t0,
        t1,
        dt0,
        initial_state,
        args=args,
        saveat=saveat,
        stepsize_controller=PIDController(
            rtol=1e-5,
            atol=1e-6,
        ),
        max_steps=int(5e6),
    )
    return solution


class SIRInitializer(Initializer):
    def __init__(self):
        super().__init__(
            description="a basic Covid SIR model initializer",
            initialize_date=date(2025, 2, 20),
            population_size=100000,
        )

    def get_initial_state(self, model: CompartmentalModel, **kwargs):
        s = model.get_compartment("s")
        i = model.get_compartment("i")

        initial_infections_proportion = kwargs["inf_prop"]
        initial_infections_count = (
            self.population_size * initial_infections_proportion
        )
        s.values = jnp.array([self.population_size - initial_infections_count])
        i.values = jnp.array([initial_infections_count])
        # r.values already set to 0.0
        return model.compartments


class CovidSIRModel(CompartmentalModel):
    def __init__(self):
        initializer = SIRInitializer()
        strains = [
            Strain(
                strain_name="X",
                r0=2.2,
                infectious_period=7.0,
            )
        ]
        compartments = [
            Compartment(
                name="s",
                dimensions=[
                    dimensions.Dimension(
                        name="num_susceptibles", bins=[bins.Bin()]
                    )
                ],
            ),
            Compartment(
                name="i",
                dimensions=[
                    dimensions.Dimension(
                        name="num_infectious", bins=[bins.Bin()]
                    )
                ],
            ),
            Compartment(
                name="r",
                dimensions=[
                    dimensions.Dimension(
                        name="num_recovered", bins=[bins.Bin()]
                    )
                ],
            ),
        ]
        parameters = ParamStore(
            strains=strains,
            strain_interactions={"x": {"x": 1.0}},
            ode_solver_rel_tolerance=1e-5,
            ode_solver_abs_tolerance=1e-6,
        )
        super().__init__(
            initializer=initializer,
            compartments=compartments,
            ode_function=sir_ode,
            parameters=parameters,
        )


# %%
model = CovidSIRModel()
# %%
model.initializer.get_initial_state(model, inf_prop=0.01)
# %%
solve_odes(
    model.initializer.get_initial_state(model, inf_prop=0.01),
    args=model.parameters,
    tf=100.0,
    model=sir_ode,
)
# %%
