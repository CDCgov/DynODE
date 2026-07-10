from __future__ import annotations

from typing import Literal

import diffrax as dfx

from .base import SolverMethodSpec


class HeunSpec(SolverMethodSpec):
    type: Literal["heun"] = "heun"

    def to_diffrax(self) -> dfx.AbstractSolver:
        return dfx.Heun()
