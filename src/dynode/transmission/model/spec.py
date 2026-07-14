from __future__ import annotations

from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from typing_extensions import Self

from dynode.transmission.interactions.spec import InteractionSpec
from dynode.transmission.strains.spec import StrainSpec

from .matrix import (
    interaction_matrix_dump as build_interaction_matrix_dump,
)
from .matrix import (
    interaction_matrix_named as build_interaction_matrix_named,
)
from .matrix import (
    interaction_matrix_spec as build_interaction_matrix_spec,
)
from .matrix import (
    interaction_value as get_interaction_value,
)
from .validation import (
    validate_interaction_targets as check_interaction_targets,
)
from .validation import (
    validate_introduction_ages_consistent as check_introduction_ages_consistent,
)
from .validation import (
    validate_optional_strain_fields_consistent as check_optional_strain_fields_consistent,
)
from .validation import (
    validate_unique_strain_names as check_unique_strain_names,
)


class TransmissionSpec(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
        validate_assignment=True,
    )

    strains: list[StrainSpec] = Field(
        min_length=1,
        description="Strains included in the transmission model.",
    )

    default_offdiag: InteractionSpec = Field(
        default_factory=lambda: InteractionSpec.deterministic("crossimmunity"),
        description=(
            "Fallback interaction used for unspecified off-diagonal "
            "strain interaction pairs."
        ),
    )

    force_diag_ones: bool = Field(
        default=True,
        description=(
            "If True, diagonal strain interactions are forced to 1.0 unless "
            "explicitly set."
        ),
    )

    @property
    def strain_names(self) -> list[str]:
        return [strain.name for strain in self.strains]

    @property
    def strains_to_idx(self) -> dict[str, int]:
        return {name: i for i, name in enumerate(self.strain_names)}

    @field_validator("strains")
    @classmethod
    def validate_non_empty_strains(
        cls,
        strains: list[StrainSpec],
    ) -> list[StrainSpec]:
        if not strains:
            raise ValueError(
                "TransmissionSpec.strains must contain at least one strain."
            )
        return strains

    @model_validator(mode="after")
    def validate_unique_strain_names(self) -> Self:
        check_unique_strain_names(self)
        return self

    @model_validator(mode="after")
    def validate_interaction_targets(self) -> Self:
        check_interaction_targets(self)
        return self

    @model_validator(mode="after")
    def validate_introduction_ages_consistent(self) -> Self:
        check_introduction_ages_consistent(self)
        return self

    @model_validator(mode="after")
    def validate_optional_strain_fields_consistent(self) -> Self:
        check_optional_strain_fields_consistent(self)
        return self

    def get_strain(self, name: str) -> StrainSpec:
        for strain in self.strains:
            if strain.name == name:
                return strain

        raise KeyError(
            f"Unknown strain {name!r}. Known strains are: {self.strain_names}."
        )

    def interaction_value(
        self,
        source_name: str,
        target_name: str,
    ) -> InteractionSpec:
        """Return the interaction spec for source -> target."""
        return get_interaction_value(
            transmission=self,
            source_name=source_name,
            target_name=target_name,
        )

    def interaction_matrix_spec(self) -> list[list[InteractionSpec]]:
        """Return an n x n matrix of strain interaction specs."""
        return build_interaction_matrix_spec(self)

    def interaction_matrix_named(
        self,
    ) -> dict[str, dict[str, InteractionSpec]]:
        """Return a readable named interaction matrix representation."""
        return build_interaction_matrix_named(self)

    def interaction_matrix_dump(self) -> list[list[dict[str, Any]]]:
        """JSON-serializable version of the interaction matrix."""
        return build_interaction_matrix_dump(self)
