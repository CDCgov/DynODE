from pydantic import (
    BaseModel,
    NonNegativeInt,
    PositiveFloat,
    model_validator,
)
from typing_extensions import Self


class CategoricalBin(BaseModel):
    "Bin with a distinct name"

    name: str


class DiscretizedPositiveIntBin(BaseModel):
    "Bin with a distinct discretized positive int inclusive min/max."

    min_value: NonNegativeInt
    max_value: NonNegativeInt

    @model_validator(mode="after")
    def bin_valid_side(self) -> Self:
        """Asserts that min_value <= max_value"""
        assert self.min_value <= self.max_value
        return self


class AgeBin(DiscretizedPositiveIntBin):
    pass


class WaneBin(CategoricalBin):
    waning_time: NonNegativeInt
    waning_protection: PositiveFloat
