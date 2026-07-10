from __future__ import annotations

from typing import Literal

import diffrax as dfx

from .base import SolverMethodSpec


class Bosh3Spec(SolverMethodSpec):
    type: Literal["bosh3"] = "bosh3"

    def to_diffrax(self) -> dfx.AbstractSolver:
        return dfx.Bosh3()
