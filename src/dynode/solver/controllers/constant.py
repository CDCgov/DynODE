from __future__ import annotations

from typing import Literal

import diffrax as dfx

from .base import StepSizeControllerSpec


class ConstantStepSizeSpec(StepSizeControllerSpec):
    """
    Fixed step-size controller.

    The actual fixed step size is supplied as SolverSpec.dt0.
    """

    type: Literal["constant"] = "constant"

    def to_diffrax(self) -> dfx.AbstractStepSizeController:
        return dfx.ConstantStepSize()
