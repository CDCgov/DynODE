from __future__ import annotations

from typing import Any

import diffrax as dfx
import jax.numpy as jnp
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveFloat,
    PositiveInt,
    model_validator,
)
from typing_extensions import Self

from .controllers.constant import ConstantStepSizeSpec
from .controllers.pid import PIDControllerSpec
from .controllers.unions import StepSizeController
from .methods.tsit5 import Tsit5Spec
from .methods.unions import SolverMethod
from .save_at import SaveAtSpec


class SolverSpec(BaseModel):
    """
    Declarative ODE solver configuration.

    This spec validates solver-related configuration and compiles it into the
    keyword arguments expected by diffrax.diffeqsolve.

    It should not:
    - hold raw Diffrax solver objects in config files
    - call diffrax.diffeqsolve itself
    - know anything about compartments, strains, priors, or data
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )

    solver: SolverMethod = Field(
        default_factory=Tsit5Spec,
        description="Numerical solver method.",
    )

    stepsize_controller: StepSizeController = Field(
        default_factory=PIDControllerSpec,
        description="Step-size controller.",
    )

    dt0: PositiveFloat | None = Field(
        default=0.1,
        description=(
            "Initial step size. Required for constant-step solves. "
            "For adaptive solves, this is the initial proposed step size."
        ),
    )

    max_steps: PositiveInt | None = Field(
        default=int(1e6),
        description=(
            "Maximum number of solver steps. Use None only with care; some "
            "Diffrax save modes are incompatible with unlimited steps."
        ),
    )

    save_at: SaveAtSpec = Field(
        default_factory=SaveAtSpec,
        description="What solution values to save.",
    )

    step_ts: tuple[float, ...] = Field(
        default_factory=tuple,
        description=(
            "Times that an adaptive solver should step to exactly. "
            "Used via ClipStepSizeController."
        ),
    )

    jump_ts: tuple[float, ...] = Field(
        default_factory=tuple,
        description=(
            "Known discontinuity times. Adaptive solvers are clipped to step "
            "around these times using ClipStepSizeController."
        ),
    )

    throw: bool = Field(
        default=True,
        description="Whether Diffrax should raise on solver failure.",
    )

    @property
    def discontinuity_points(self) -> tuple[float, ...]:
        """
        Backward-compatible naming for the old SolverParams field.
        """
        return self.jump_ts

    @model_validator(mode="after")
    def validate_solver_spec(self) -> Self:
        self._validate_times("step_ts", self.step_ts)
        self._validate_times("jump_ts", self.jump_ts)
        self._validate_controller_compatibility()
        self._validate_max_steps_saveat_compatibility()
        return self

    def _validate_times(
        self,
        field_name: str,
        values: tuple[float, ...],
    ) -> None:
        if any(value < 0 for value in values):
            raise ValueError(f"{field_name} must contain non-negative times.")

        if any(b <= a for a, b in zip(values, values[1:])):
            raise ValueError(f"{field_name} must be strictly increasing.")

    def _validate_controller_compatibility(self) -> None:
        if isinstance(self.stepsize_controller, ConstantStepSizeSpec):
            if self.dt0 is None:
                raise ValueError(
                    "dt0 is required when using ConstantStepSizeSpec."
                )

            if self.step_ts or self.jump_ts:
                raise ValueError(
                    "step_ts and jump_ts require an adaptive step-size controller. "
                    "Use PIDControllerSpec if you need exact step or jump times."
                )

        if isinstance(self.stepsize_controller, PIDControllerSpec):
            if self.dt0 is not None and self.dt0 <= 0:
                raise ValueError("dt0 must be positive when provided.")

    def _validate_max_steps_saveat_compatibility(self) -> None:
        if self.max_steps is not None:
            return

        if self.save_at.dense:
            raise ValueError(
                "max_steps=None is not allowed with save_at.dense=True."
            )

        if self.save_at.steps:
            raise ValueError(
                "max_steps=None is not allowed when saving solver steps."
            )

    def solver_method(self) -> dfx.AbstractSolver:
        return self.solver.to_diffrax()

    def controller(self) -> dfx.AbstractStepSizeController:
        controller = self.stepsize_controller.to_diffrax()

        if self.step_ts or self.jump_ts:
            if not self.stepsize_controller.is_adaptive:
                raise ValueError(
                    "step_ts and jump_ts require an adaptive step-size controller."
                )

            controller = dfx.ClipStepSizeController(
                controller,
                step_ts=None
                if not self.step_ts
                else jnp.asarray(self.step_ts),
                jump_ts=None
                if not self.jump_ts
                else jnp.asarray(self.jump_ts),
            )

        return controller

    def saveat(self) -> dfx.SaveAt:
        return self.save_at.to_diffrax()

    def diffeqsolve_kwargs(self) -> dict[str, Any]:
        """
        Keyword arguments intended for diffrax.diffeqsolve.

        Runtime code can do:

            sol = dfx.diffeqsolve(
                terms,
                t0=t0,
                t1=t1,
                y0=y0,
                args=args,
                **solver_spec.diffeqsolve_kwargs(),
            )
        """
        return {
            "solver": self.solver_method(),
            "dt0": self.dt0,
            "stepsize_controller": self.controller(),
            "saveat": self.saveat(),
            "max_steps": self.max_steps,
            "throw": self.throw,
        }
