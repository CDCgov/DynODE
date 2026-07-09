from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from typing_extensions import Self

from .interaction_spec import InteractionSpec
from .strain_spec import StrainSpec


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
        description="If True, diagonal strain interactions are forced to 1.0 unless explicitly set.",
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
        names = self.strain_names
        duplicates = sorted({name for name in names if names.count(name) > 1})

        if duplicates:
            raise ValueError(f"Duplicate strain names found: {duplicates}")

        return self

    @model_validator(mode="after")
    def validate_interaction_targets(self) -> Self:
        valid_names = set(self.strain_names)

        for strain in self.strains:
            unknown_targets = set(strain.interactions) - valid_names

            if unknown_targets:
                raise ValueError(
                    f"Strain {strain.name!r} defines interactions for unknown "
                    f"target strains: {sorted(unknown_targets)}. "
                    f"Known strains are: {sorted(valid_names)}."
                )

        return self

    @model_validator(mode="after")
    def validate_introduction_ages_consistent(self) -> Self:
        introduced = [
            strain
            for strain in self.strains
            if getattr(strain, "is_introduced", False)
        ]

        intro_age_sets = [
            tuple(strain.introduction_ages)
            for strain in introduced
            if getattr(strain, "introduction_ages", None) is not None
        ]

        if intro_age_sets:
            first = intro_age_sets[0]
            mismatched = [
                strain.name
                for strain in introduced
                if getattr(strain, "introduction_ages", None) is not None
                and tuple(strain.introduction_ages) != first
            ]

            if mismatched:
                raise ValueError(
                    "Currently all introduced strains must have matching "
                    f"introduction_ages. Mismatched strains: {mismatched}"
                )

        return self

    @model_validator(mode="after")
    def validate_optional_strain_fields_consistent(self) -> Self:
        """
        If a strain-level optional field is set for one strain, require it for all.
        Add/remove names here as your StrainSpec evolves.
        """
        optional_fields_to_check = [
            "exposed_to_infectious",
            "vaccine_efficacy",
        ]

        for field_name in optional_fields_to_check:
            present = [
                strain.name
                for strain in self.strains
                if getattr(strain, field_name, None) is not None
            ]

            if present and len(present) != len(self.strains):
                missing = [
                    strain.name
                    for strain in self.strains
                    if getattr(strain, field_name, None) is None
                ]

                raise ValueError(
                    f"If {field_name!r} is set for one strain, it must be set "
                    f"for all strains. Present={present}, missing={missing}."
                )

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
        """
        Return the interaction spec for source -> target.

        Resolution order:
        1. Explicit source.interactions[target]
        2. Diagonal default, if source == target
        3. default_offdiag
        """
        source = self.get_strain(source_name)

        if target_name in source.interactions:
            return source.interactions[target_name]

        if source_name == target_name and self.force_diag_ones:
            return InteractionSpec.fixed(1.0)

        return self.default_offdiag

    def interaction_matrix_spec(self) -> list[list[InteractionSpec]]:
        """
        Return an n x n matrix where matrix[i][j] is the effect of source strain i
        on target strain j.
        """
        names = self.strain_names

        return [
            [
                self.interaction_value(
                    source_name=source_name,
                    target_name=target_name,
                )
                for target_name in names
            ]
            for source_name in names
        ]

    def interaction_matrix_named(
        self,
    ) -> dict[str, dict[str, InteractionSpec]]:
        """
        More readable named representation.

        Useful for debugging, serialization, and tests.
        """
        names = self.strain_names

        return {
            source_name: {
                target_name: self.interaction_value(source_name, target_name)
                for target_name in names
            }
            for source_name in names
        }

    def interaction_matrix_dump(self) -> list[list[dict[str, Any]]]:
        """
        JSON-serializable version of the interaction matrix.
        """
        return [
            [value.model_dump() for value in row]
            for row in self.interaction_matrix_spec()
        ]
