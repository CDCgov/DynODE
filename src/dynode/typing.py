from __future__ import annotations

from typing import Annotated

from pydantic import Field

DynodeName = Annotated[
    str, Field(min_length=1, pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
]
UnitIntervalFloat = Annotated[float, Field(ge=0.0, le=1.0)]
