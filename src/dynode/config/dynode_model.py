from typing import Any, Callable

from pydantic import BaseModel, ConfigDict


class DynodeModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    spec: ModelSpec
    ode: Callable[..., Any]
    observe_fn: Callable[..., Any]

    def compile(self) -> RuntimeModel:
        return compile_model(self.spec)

    def make_numpyro_model(self):
        runtime = self.compile()

        def numpyro_model(data):
            params = runtime.sample_parameters()
            y0 = runtime.initial_state(params, data)
            states = runtime.solve(self.rhs_fn, y0, params)
            self.observe_fn(states, params, data)

        return numpyro_model
