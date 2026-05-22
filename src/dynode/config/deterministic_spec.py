from pydantic import BaseModel


class DeterministicSpec(BaseModel):
    name: str
    # expression: DeterministicParameter
