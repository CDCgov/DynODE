from __future__ import annotations

from typing import Annotated

from pydantic import Field

DoseCount = Annotated[int, Field(ge=0)]
Probability = Annotated[float, Field(ge=0.0, le=1.0)]
