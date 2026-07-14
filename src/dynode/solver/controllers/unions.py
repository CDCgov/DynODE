from __future__ import annotations

from typing import Annotated

from pydantic import Field

from .constant import ConstantStepSizeSpec
from .pid import PIDControllerSpec

StepSizeController = Annotated[
    ConstantStepSizeSpec | PIDControllerSpec,
    Field(discriminator="type"),
]
