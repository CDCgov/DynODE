from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from dynode.data.data_spec import DataSpec
from dynode.parameters.parameter_spec import ParameterBlockSpec
from dynode.structure.model.model_spec import ModelSpec


class ModelInstanceSpec(BaseModel):
    """One named instance of a ModelSpec within a larger experiment."""

    model_config = ConfigDict(
        extra="forbid", frozen=True, arbitrary_types_allowed=True
    )

    key: str
    model: ModelSpec
    parameters: ParameterBlockSpec = Field(default_factory=ParameterBlockSpec)
    data: DataSpec | None = None
    static_context: dict[str, Any] = Field(default_factory=dict)
    t0: float | None = None
    t1: float | None = None
    metadata: dict[str, str] = Field(default_factory=dict)

    @property
    def local_parameter_names(self) -> set[str]:
        return self.parameters.local_parameter_names
