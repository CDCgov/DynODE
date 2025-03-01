from typing import Optional, Sequence

from jax import Array

from dynode.model_configuration import CompartmentalModel

from .params import _TransmissionParamsChex


class ODEBase:
    def __init__(self, compartmental_model: CompartmentalModel):
        pass

    def __call__(
        self,
        compartments: Sequence[Array],
        t: float,
        parameters: Optional[dict | _TransmissionParamsChex] = {},
    ) -> Sequence[Array]:
        pass


class SEIPModelOne(ODEBase):
    def __init__(self):
        super().__init__()

    def __call__(
        self,
        compartments: Sequence[Array],
        t: float,
        parameters: Optional[dict | _TransmissionParamsChex] = {},
    ) -> Sequence[Array]:
        raise NotImplementedError()
