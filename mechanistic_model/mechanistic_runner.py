"""
The following is a class which runs a series of ODE equations, performs inference, and returns Solution objects for analysis.
"""

import jax
import jax.numpy as jnp
import numpyro
from diffrax import (  # Solution,
    ConstantStepSize,
    ODETerm,
    PIDController,
    SaveAt,
    Tsit5,
    diffeqsolve,
)

numpyro.set_host_device_count(4)
jax.config.update("jax_enable_x64", True)


class MechanisticRunner:
    def __init__(self, model):
        self.model = model

    def run(self, initial_state, args, tf: int = 100):
        term = ODETerm(
            lambda t, state, parameters: self.model(state, t, parameters)
        )
        solver = Tsit5()
        t0 = 0.0
        dt0 = 1.0
        saveat = SaveAt(ts=jnp.linspace(t0, tf, int(tf) + 1))
        # jump_ts describe points in time where the model is not fully differentiable
        # this is often due to piecewise changes in parameter values like Beta
        # this is why many functions in the runner/params are required to be continuously differentiable.
        if "CONSTANT_STEP_SIZE" in args.keys() and args["CONSTANT_STEP_SIZE"]:
            dt0 = args["CONSTANT_STEP_SIZE"]
            # if user specifies they want constant step size, set it here
            stepsize_controller = ConstantStepSize()
            print("using Constant Step Size ODES with size %s" % (str(dt0)))
        else:  # otherwise use adaptive step size.
            jump_ts = (
                list(args["BETA_TIMES"])
                if "BETA_TIMES" in args.keys()
                else None
            )
            stepsize_controller = PIDController(
                rtol=1e-5,
                atol=1e-6,
                jump_ts=jump_ts,
            )

        solution = diffeqsolve(
            term,
            solver,
            t0,
            tf,
            dt0,
            initial_state,
            args=args,
            stepsize_controller=stepsize_controller,
            saveat=saveat,
            # higher for large time scales / rapid changes
            max_steps=int(1e6),
        )
        return solution
