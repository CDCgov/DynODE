import copy
import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp
from mechanistic_compartments import (
    BasicMechanisticModel,
)
from model_odes.seir_model_v5 import seirw_ode

from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5


def infer_model(times, incidence, model: BasicMechanisticModel):
    # m = model
    m = copy.deepcopy(model)
    # Parameters
    # with numpyro.plate("num_strains", model.NUM_STRAINS):
    #     excess_r0 = numpyro.sample("excess_r0", dist.Exponential(1.0))
    #     r0 = numpyro.deterministic("r0", 1 + excess_r0)
    excess_r0 = numpyro.sample("excess_r0", dist.Exponential(1.0))
    r0 = numpyro.deterministic("r0", 1 + excess_r0)

    # infectious_period = numpyro.sample(
    #     "infectious_period", dist.HalfCauchy(1.0)
    # )
    e_to_i = numpyro.sample("e_to_i", dist.HalfCauchy(1.0))

    m.STRAIN_SPECIFIC_R0 = jnp.asarray([1.5, 2.5, r0])
    # m.INFECTIOUS_PERIOD = infectious_period
    m.EXPOSED_TO_INFECTIOUS = e_to_i

    sol = m.run(
        seirw_ode, tf=100, plot=False, save=False
    )  # save_path="output/example.png")
    model_incidence = jnp.sum(sol.ys[5], axis=2)
    model_incidence = jnp.diff(model_incidence, axis=0)

    numpyro.sample("incidence", dist.Poisson(model_incidence), obs=incidence)


def infer_model2(times, incidence, initial_state, args):
    tf = 100
    new_args = dict(args)
    infectious_period = numpyro.sample(
        "infectious_period", dist.HalfCauchy(1.0)
    )
    e_to_i = numpyro.sample("e_to_i", dist.HalfCauchy(1.0))

    new_args["beta"] = new_args["beta"] * 7.0 / infectious_period
    new_args["gamma"] = new_args["gamma"] * 7.0 / infectious_period
    new_args["sigma"] = new_args["sigma"] * 3.6 / e_to_i

    term = ODETerm(
        lambda t, state, parameters: seirw_ode(state, t, parameters)
    )
    solver = Tsit5()
    t0 = 0.0
    dt0 = 0.1
    saveat = SaveAt(ts=jnp.linspace(t0, tf, int(tf) + 1))
    sol = diffeqsolve(
        term,
        solver,
        t0,
        tf,
        dt0,
        initial_state,
        args=new_args,
        saveat=saveat,
        max_steps=30000,
    )

    model_incidence = jnp.sum(sol.ys[5], axis=2)
    model_incidence = jnp.diff(model_incidence, axis=0)

    numpyro.sample("incidence", dist.Poisson(model_incidence), obs=incidence)
