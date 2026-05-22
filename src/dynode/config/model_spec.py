from pydantic import BaseModel, ConfigDict

from . import (
    DataSpec,
    ParameterSpec,
    SimulationSpec,
)


class ModelSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    name: str
    simulation: SimulationSpec
    parameters: ParameterSpec
    data: DataSpec | None = None
