from __future__ import annotations

from typing import Literal

import diffrax as dfx

from .base import SolverMethodSpec


class Kvaerno3Spec(SolverMethodSpec):
    """
    Implicit solver useful for some stiff problems.
    """

    type: Literal["kvaerno3"] = "kvaerno3"

    def to_diffrax(self) -> dfx.AbstractSolver:
        return dfx.Kvaerno3()
