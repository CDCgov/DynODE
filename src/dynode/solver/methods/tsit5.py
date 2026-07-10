from __future__ import annotations

from typing import Literal

import diffrax as dfx

from .base import SolverMethodSpec


class Tsit5Spec(SolverMethodSpec):
    type: Literal["tsit5"] = "tsit5"

    def to_diffrax(self) -> dfx.AbstractSolver:
        return dfx.Tsit5()
