from pydantic import (
    BaseModel,
    ConfigDict,
)


class ParameterSet(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
