from pydantic import BaseModel

from . import DistributionSpec


class PriorSpec(BaseModel):
    name: str
    distribution: DistributionSpec
