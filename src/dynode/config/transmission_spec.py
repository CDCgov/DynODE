from pydantic import BaseModel, ConfigDict

from . import InteractionSpec, StrainSpec


class TransmissionSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    strains: list[StrainSpec]
    default_offdiag: InteractionSpec = InteractionSpec.deterministic(
        "crossimmunity"
    )
    force_diag_ones: bool = True

    def interaction_matrix_spec(self):
        pass
