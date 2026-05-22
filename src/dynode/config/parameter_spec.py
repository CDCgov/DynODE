from pydantic import BaseModel

from . import (
    DeterministicSpec,
    PriorSpec,
    SolverSpec,
    TransmissionSpec,
)


class ParameterSpec(BaseModel):
    solver: SolverSpec
    transmission: TransmissionSpec
    priors: list[PriorSpec]
    deterministic: list[DeterministicSpec]
