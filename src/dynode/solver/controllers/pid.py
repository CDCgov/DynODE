from __future__ import annotations

from typing import Literal

import diffrax as dfx
from pydantic import Field, NonNegativeFloat, PositiveFloat

from .base import StepSizeControllerSpec


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
