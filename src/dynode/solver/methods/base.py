from __future__ import annotations

from abc import ABC, abstractmethod

import diffrax as dfx
from pydantic import BaseModel, ConfigDict


class SolverMethodSpec(BaseModel, ABC):
    """
    Declarative Diffrax solver-method spec.

    This is intentionally serializable. Do not store raw Diffrax solver
    objects directly in model configs.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )

    type: str

    @abstractmethod
    def to_diffrax(self) -> dfx.AbstractSolver:
        raise NotImplementedError
