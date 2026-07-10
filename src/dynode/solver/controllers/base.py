from __future__ import annotations

from abc import ABC, abstractmethod

import diffrax as dfx
from pydantic import BaseModel, ConfigDict


class StepSizeControllerSpec(BaseModel, ABC):
    """
    Declarative Diffrax step-size-controller spec.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )

    type: str

    @abstractmethod
    def to_diffrax(self) -> dfx.AbstractStepSizeController:
        raise NotImplementedError

    @property
    def is_adaptive(self) -> bool:
        return False
