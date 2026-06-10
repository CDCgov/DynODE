from pydantic import BaseModel

from . import DistributionSpec


# Need to add the union type alias above PriorSpec and change distribution to be that type alias
class PriorSpec(BaseModel):
    name: str
    distribution: DistributionSpec
