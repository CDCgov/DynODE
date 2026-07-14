from __future__ import annotations

from typing import Annotated

from pydantic import Field

from .age import AgeBin
from .base import BinSpec
from .discretized import DiscretizedPositiveIntBin
from .wane import WaneBin

AnyBinSpec = Annotated[
    BinSpec | DiscretizedPositiveIntBin | AgeBin | WaneBin,
    Field(discriminator="type"),
]
