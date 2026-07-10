from __future__ import annotations

from typing import Literal

import diffrax as dfx

from .base import SolverMethodSpec


class Kvaerno5Spec(SolverMethodSpec):
    type: Literal["kvaerno5"] = "kvaerno5"

    def to_diffrax(self) -> dfx.AbstractSolver:
        return dfx.Kvaerno5()
