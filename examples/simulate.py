# An Example of how to simulate a basic SIR compartmental model using the dynode package
# Including all the class setup
# %% imports, mostly for class creation
from datetime import date

import chex
import matplotlib.pyplot as plt
from jax import Array

from dynode.model_configuration import (
    Compartment,
    Initializer,
    SimulationConfig,
)
from dynode.model_configuration.bins import Bin
from dynode.model_configuration.dimension import Dimension
from dynode.model_configuration.params import (
    Params,
    SolverParams,
    TransmissionParams,
)
from dynode.model_configuration.strains import Strain
from dynode.odes import AbstractODEParams, ODEBase

# %% class definitions


class SIRInitializer(Initializer):
    def __init__(self):
        super().__init__(
            description="An SIR initalizer",
            initialize_date=date(2025, 3, 13),
            population_size=1,
        )

    def get_initial_state(self, **kwargs):
        config: SimulationConfig = kwargs["SIRConfig"]
        s = config.get_compartment("s")
        i = config.get_compartment("i")
        r = config.get_compartment("r")
        s.values = 0.99
        i.values = 0.01
        return [s, i, r]


class SIRConfig(SimulationConfig):
    def __init__(self):
        dimension = Dimension(name="value", bins=[Bin(name="value")])
        s = Compartment(name="s", dimensions=[dimension])
        i = Compartment(name="i", dimensions=[dimension])
        r = Compartment(name="r", dimensions=[dimension])
        strain = [
            Strain(strain_name="example_strain", r0=2.0, infectious_period=7.0)
        ]
        parameters = Params(
            solver_params=SolverParams(),
            transmission_params=TransmissionParams(
                strains=strain,
                strain_interactions={
                    "example_strain": {"example_strain": 1.0}
                },
            ),
        )
        super().__init__(
            compartments=[s, i, r],
            initializer=SIRInitializer(),
            parameters=parameters,
        )


@chex.dataclass
class SIR_ODEParams(AbstractODEParams):
    beta: chex.ArrayDevice  # r0/infectious period
    sigma: chex.ArrayDevice  # 1/infectious period
    pass


class SIR_ODE(ODEBase):
    def __call__(self, t: float, y: tuple[Array], params: SIR_ODEParams):
        s, i, _ = y
        beta = params.beta
        ds = -beta * s * i
        dr = i * params.sigma
        return [ds, -ds, dr]


# %% simulation

# set up config
config = SIRConfig()


def get_odeparams(transmission_params: TransmissionParams):
    strain = transmission_params.strains[0]
    beta = strain.r0 / strain.infectious_period
    sigma = 1 / strain.infectious_period
    return SIR_ODEParams(beta=beta, sigma=sigma)


ode_params = get_odeparams(config.parameters.transmission_params)

# set up odes
ode = SIR_ODE()
# we need just the jax arrays for the initial state to the ODEs
intial_state = [
    compartment.values
    for compartment in config.initializer.get_initial_state(SIRConfig=config)
]

# solve the odes for 100 days
solution = ode.solve(
    initial_state=intial_state,
    solver_parameters=config.parameters.solver_params,
    ode_parameters=ode_params,
    duration_days=100,
)

# %% plot
plt.plot(solution.ys[0], label="s")
plt.plot(solution.ys[1], label="i")
plt.plot(solution.ys[2], label="r")
plt.show()
