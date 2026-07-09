from __future__ import annotations

from typing import Annotated

from pydantic import Field

from .binary import BinaryValueSpec
from .constant import ConstantValueSpec
from .function import FunctionValueSpec
from .references import DataRef, DeterministicRef, ParamRef
from .unary import UnaryValueSpec

ValueExpression = Annotated[
    ConstantValueSpec
    | ParamRef
    | DeterministicRef
    | DataRef
    | UnaryValueSpec
    | BinaryValueSpec
    | FunctionValueSpec,
    Field(discriminator="type"),
]

for _model in (
    UnaryValueSpec,
    BinaryValueSpec,
    FunctionValueSpec,
):
    _model.model_rebuild(_types_namespace={"ValueExpression": ValueExpression})

DistributionValue = ValueExpression
DeterministicExpression = ValueExpression
InitializerValue = ValueExpression
InteractionValue = ValueExpression
ParameterValue = ValueExpression
