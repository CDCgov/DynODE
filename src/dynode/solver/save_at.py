from __future__ import annotations

import diffrax as dfx
import jax.numpy as jnp
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator
from typing_extensions import Self


class SaveAtSpec(BaseModel):
    """
    Declarative Diffrax SaveAt spec.

    This controls what solution values Diffrax saves.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )

    t0: bool = Field(
        default=False,
        description="Save the initial state.",
    )
    t1: bool = Field(
        default=True,
        description="Save the final state.",
    )
    ts: tuple[float, ...] | None = Field(
        default=None,
        description="Specific times at which to save the solution.",
    )
    dense: bool = Field(
        default=False,
        description="Whether to save dense output.",
    )
    steps: bool | PositiveInt = Field(
        default=False,
        description="Save every nth solver step if an integer is provided.",
    )

    @model_validator(mode="after")
    def validate_saveat(self) -> Self:
        if self.ts is not None:
            if len(self.ts) == 0:
                raise ValueError("save_at.ts cannot be empty.")

            if any(b <= a for a, b in zip(self.ts, self.ts[1:])):
                raise ValueError("save_at.ts must be strictly increasing.")

        return self

    def to_diffrax(self) -> dfx.SaveAt:
        return dfx.SaveAt(
            t0=self.t0,
            t1=self.t1,
            ts=None if self.ts is None else jnp.asarray(self.ts),
            dense=self.dense,
            steps=self.steps,
        )
