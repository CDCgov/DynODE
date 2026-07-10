from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from dynode.typing import DynodeName


class BinSpec(BaseModel):
    """
    Base declarative bin specification.

    A bin is one named cell within a DimensionSpec.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )

    type: Literal["generic"] = Field(
        default="generic",
        description="Bin type discriminator.",
    )

    name: DynodeName = Field(
        description=(
            "Bin name. Must be unique within a DimensionSpec. "
            "Should be safe to use as an attribute name."
        ),
    )

    description: str | None = Field(
        default=None,
        description="Optional human-readable description.",
    )

    metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Optional metadata for documentation, UI display, or auditing.",
    )

    def contains(self, value: Any) -> bool:
        """
        Whether this bin contains a value.

        Generic categorical bins only match by name. Numeric subclasses
        override this method.
        """
        return value == self.name
