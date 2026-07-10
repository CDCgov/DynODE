from __future__ import annotations

from typing import Literal

import diffrax as dfx

from .base import SolverMethodSpec


class EulerSpec(SolverMethodSpec):
    type: Literal["euler"] = "euler"

    def to_diffrax(self) -> dfx.AbstractSolver:
        return dfx.Euler()
