from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Annotated, Any, Literal

import diffrax as dfx
import jax.numpy as jnp
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
    model_validator,
)
from typing_extensions import Self


class SolverMethodSpec(BaseModel, ABC):
    """
    Declarative Diffrax solver-method spec.

    This is intentionally serializable. Do not store raw Diffrax solver
    objects directly in model configs.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )

    type: str

    @abstractmethod
    def to_diffrax(self) -> dfx.AbstractSolver:
        raise NotImplementedError


class Tsit5Spec(SolverMethodSpec):
    type: Literal["tsit5"] = "tsit5"

    def to_diffrax(self) -> dfx.AbstractSolver:
        return dfx.Tsit5()


class Dopri5Spec(SolverMethodSpec):
    type: Literal["dopri5"] = "dopri5"

    def to_diffrax(self) -> dfx.AbstractSolver:
        return dfx.Dopri5()


class Dopri8Spec(SolverMethodSpec):
    type: Literal["dopri8"] = "dopri8"

    def to_diffrax(self) -> dfx.AbstractSolver:
        return dfx.Dopri8()


class Bosh3Spec(SolverMethodSpec):
    type: Literal["bosh3"] = "bosh3"

    def to_diffrax(self) -> dfx.AbstractSolver:
        return dfx.Bosh3()


class EulerSpec(SolverMethodSpec):
    type: Literal["euler"] = "euler"

    def to_diffrax(self) -> dfx.AbstractSolver:
        return dfx.Euler()


class HeunSpec(SolverMethodSpec):
    type: Literal["heun"] = "heun"

    def to_diffrax(self) -> dfx.AbstractSolver:
        return dfx.Heun()


class Kvaerno3Spec(SolverMethodSpec):
    """
    Implicit solver useful for some stiff problems.
    """

    type: Literal["kvaerno3"] = "kvaerno3"

    def to_diffrax(self) -> dfx.AbstractSolver:
        return dfx.Kvaerno3()


class Kvaerno4Spec(SolverMethodSpec):
    type: Literal["kvaerno4"] = "kvaerno4"

    def to_diffrax(self) -> dfx.AbstractSolver:
        return dfx.Kvaerno4()


class Kvaerno5Spec(SolverMethodSpec):
    type: Literal["kvaerno5"] = "kvaerno5"

    def to_diffrax(self) -> dfx.AbstractSolver:
        return dfx.Kvaerno5()


SolverMethod = Annotated[
    Tsit5Spec
    | Dopri5Spec
    | Dopri8Spec
    | Bosh3Spec
    | EulerSpec
    | HeunSpec
    | Kvaerno3Spec
    | Kvaerno4Spec
    | Kvaerno5Spec,
    Field(discriminator="type"),
]


class StepSizeControllerSpec(BaseModel, ABC):
    """
    Declarative Diffrax step-size-controller spec.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )

    type: str

    @abstractmethod
    def to_diffrax(self) -> dfx.AbstractStepSizeController:
        raise NotImplementedError

    @property
    def is_adaptive(self) -> bool:
        return False


class ConstantStepSizeSpec(StepSizeControllerSpec):
    """
    Fixed step-size controller.

    The actual fixed step size is supplied as SolverSpec.dt0.
    """

    type: Literal["constant"] = "constant"

    def to_diffrax(self) -> dfx.AbstractStepSizeController:
        return dfx.ConstantStepSize()


class PIDControllerSpec(StepSizeControllerSpec):
    """
    Adaptive step-size controller.

    Diffrax uses rtol and atol to control local error. Optional PID coefficients
    are exposed for advanced tuning.
    """

    type: Literal["pid"] = "pid"

    rtol: PositiveFloat = Field(
        default=1e-5,
        description="Relative tolerance for adaptive stepping.",
    )
    atol: PositiveFloat = Field(
        default=1e-6,
        description="Absolute tolerance for adaptive stepping.",
    )
    pcoeff: NonNegativeFloat = Field(
        default=0.0,
        description="Proportional coefficient for PID control.",
    )
    icoeff: NonNegativeFloat = Field(
        default=1.0,
        description="Integral coefficient for PID control.",
    )
    dcoeff: NonNegativeFloat = Field(
        default=0.0,
        description="Derivative coefficient for PID control.",
    )
    safety: PositiveFloat = Field(
        default=0.9,
        description="Safety factor for adaptive step-size changes.",
    )

    @property
    def is_adaptive(self) -> bool:
        return True

    def to_diffrax(self) -> dfx.AbstractStepSizeController:
        return dfx.PIDController(
            rtol=self.rtol,
            atol=self.atol,
            pcoeff=self.pcoeff,
            icoeff=self.icoeff,
            dcoeff=self.dcoeff,
            safety=self.safety,
        )


StepSizeController = Annotated[
    ConstantStepSizeSpec | PIDControllerSpec,
    Field(discriminator="type"),
]


class SaveAtSpec(BaseModel):
    """
    Declarative Diffrax SaveAt spec.

    This controls what solution values Diffrax saves.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )

    t0: bool = Field(
        default=False,
        description="Save the initial state.",
    )
    t1: bool = Field(
        default=True,
        description="Save the final state.",
    )
    ts: tuple[float, ...] | None = Field(
        default=None,
        description="Specific times at which to save the solution.",
    )
    dense: bool = Field(
        default=False,
        description="Whether to save dense output.",
    )
    steps: bool | PositiveInt = Field(
        default=False,
        description="Save every nth solver step if an integer is provided.",
    )

    @model_validator(mode="after")
    def validate_saveat(self) -> Self:
        if self.ts is not None:
            if len(self.ts) == 0:
                raise ValueError("save_at.ts cannot be empty.")

            if any(b <= a for a, b in zip(self.ts, self.ts[1:])):
                raise ValueError("save_at.ts must be strictly increasing.")

        return self

    def to_diffrax(self) -> dfx.SaveAt:
        return dfx.SaveAt(
            t0=self.t0,
            t1=self.t1,
            ts=None if self.ts is None else jnp.asarray(self.ts),
            dense=self.dense,
            steps=self.steps,
        )


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

    # Backward-compatible naming for your old SolverParams field.
    @property
    def discontinuity_points(self) -> tuple[float, ...]:
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
