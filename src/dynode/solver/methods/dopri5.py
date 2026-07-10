from __future__ import annotations

from typing import Literal

import diffrax as dfx

from .base import SolverMethodSpec


class Dopri5Spec(SolverMethodSpec):
    type: Literal["dopri5"] = "dopri5"

    def to_diffrax(self) -> dfx.AbstractSolver:
        return dfx.Dopri5()
