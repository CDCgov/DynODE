from pydantic import BaseModel, ConfigDict, model_validator

from . import CompartmentSpec, InitializerSpec


class SimulationSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    initializer: InitializerSpec
    compartments: list[CompartmentSpec]

    @model_validator(mode="after")
    def validate_compartments(self):
        return self
