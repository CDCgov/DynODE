from __future__ import annotations

from typing import Annotated

from pydantic import Field

from .bosh3 import Bosh3Spec
from .dopri5 import Dopri5Spec
from .dopri8 import Dopri8Spec
from .euler import EulerSpec
from .heun import HeunSpec
from .kvaerno3 import Kvaerno3Spec
from .kvaerno4 import Kvaerno4Spec
from .kvaerno5 import Kvaerno5Spec
from .tsit5 import Tsit5Spec


SolverMethod = Annotated[
    Tsit5Spec
    | Dopri5Spec
    | Dopri8Spec
    | Bosh3Spec
    | EulerSpec
    | HeunSpec
    | Kvaerno3Spec
    | Kvaerno4Spec
    | Kvaerno5Spec,
    Field(discriminator="type"),
]
