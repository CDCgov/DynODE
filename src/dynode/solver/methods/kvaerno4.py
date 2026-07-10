from __future__ import annotations

from typing import Literal

import diffrax as dfx

from .base import SolverMethodSpec


class Kvaerno4Spec(SolverMethodSpec):
    type: Literal["kvaerno4"] = "kvaerno4"

    def to_diffrax(self) -> dfx.AbstractSolver:
        return dfx.Kvaerno4()
