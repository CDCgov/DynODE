from __future__ import annotations

from typing import Literal

import diffrax as dfx

from .base import SolverMethodSpec


class Dopri8Spec(SolverMethodSpec):
    type: Literal["dopri8"] = "dopri8"

    def to_diffrax(self) -> dfx.AbstractSolver:
        return dfx.Dopri8()
