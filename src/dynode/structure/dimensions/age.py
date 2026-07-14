from __future__ import annotations

from typing import Literal

from pydantic import Field

from dynode.structure.bins.age import AgeBin
from dynode.typing import DynodeName

from .base import DimensionSpec


class AgeDimensionSpec(DimensionSpec):
    """
    Dimension for age stratification.
    """

    type: Literal["age"] = "age"

    name: DynodeName = Field(
        default="age",
        description="Age dimension name.",
    )

    bins: tuple[AgeBin, ...] = Field(
        min_length=1,
        description="Age bins.",
    )
