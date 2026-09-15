from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from dynode.structure.bins import AgeBin
from dynode.transmission.interactions.spec import InteractionSpec
from dynode.typing import DynodeName
from dynode.value.coercion import coerce_value_fields
from dynode.value.unions import ParameterValue

from .dependencies import (
    strain_data_dependencies,
    strain_deterministic_dependencies,
    strain_parameter_dependencies,
    strain_parameter_values,
)
from .evaluation import evaluated_strain_values
from .introduction import introduction_age_mask as build_introduction_age_mask
from .types import DoseCount, Probability
from .validation import (
    validate_core_values,
    validate_introduction_fields,
    validate_no_data_dependencies,
)


class StrainSpec(BaseModel):
    """
    Declarative strain specification.

    Strain-level parameter fields use the shared value system, so they may be
    constants, parameter references, deterministic references, or expressions.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )

    name: DynodeName = Field(
        description="Strain name, no leading numbers or special characters."
    )

    r0: ParameterValue = Field(
        description=(
            "Strain reproduction number used to calculate transmission rate. "
            "Must be non-negative if provided as a constant."
        ),
    )

    infectious_period: ParameterValue = Field(
        description=(
            "Average number of days a freshly infectious population stays "
            "infectious. Must be positive if provided as a constant."
        ),
    )

    exposed_to_infectious: ParameterValue | None = Field(
        default=None,
        description=(
            "Average number of days between exposure to this strain and "
            "becoming infectious. Must be positive if provided."
        ),
    )

    vaccine_efficacy: dict[DoseCount, Probability] | None = Field(
        default=None,
        description=(
            "Maps tracked vaccine dose count to protection against infection "
            "from this strain before immune waning. Values must be in [0, 1]."
        ),
    )

    interactions: dict[str, InteractionSpec] = Field(
        default_factory=dict,
        description=(
            "Explicit source-strain to target-strain interaction specs. "
            "Keys are target strain names. TransmissionSpec validates that "
            "the target names exist."
        ),
    )

    is_introduced: bool = Field(
        default=False,
        description=(
            "Whether this strain is introduced from an external population "
            "during simulation."
        ),
    )

    introduction_time: ParameterValue | None = Field(
        default=None,
        description=(
            "Date or non-negative simulation time of peak external infectious "
            "population mixing. Only used if is_introduced is True."
        ),
    )

    introduction_percentage: ParameterValue | None = Field(
        default=None,
        description=(
            "External population size relative to tracked population. "
            "Must be positive if provided as a constant. "
            "Only used if is_introduced is True."
        ),
    )

    introduction_scale: ParameterValue | None = Field(
        default=None,
        description=(
            "Spread of external infectious population mixing around "
            "introduction_time. Must be positive if provided as a constant. "
            "Only used if is_introduced is True."
        ),
    )

    introduction_ages: tuple[AgeBin, ...] | None = Field(
        default=None,
        description=(
            "Age bins receiving external introductions. "
            "Only used if is_introduced is True."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def coerce_parameter_values(cls, data: Any) -> Any:
        """Allow simple Python/YAML forms like r0: 2.5."""
        return coerce_value_fields(
            data,
            (
                "r0",
                "infectious_period",
                "exposed_to_infectious",
                "introduction_time",
                "introduction_percentage",
                "introduction_scale",
            ),
        )

    @model_validator(mode="after")
    def validate_strain_spec(self) -> Self:
        validate_core_values(self)
        validate_introduction_fields(self)
        validate_no_data_dependencies(
            strain_name=str(self.name),
            data_dependencies=self.data_dependencies,
        )
        return self

    @property
    def parameter_dependencies(self) -> set[str]:
        return strain_parameter_dependencies(self)

    @property
    def deterministic_dependencies(self) -> set[str]:
        return strain_deterministic_dependencies(self)

    @property
    def dependencies(self) -> set[str]:
        """All parameter-like dependencies needed to evaluate this strain."""
        return self.parameter_dependencies | self.deterministic_dependencies

    @property
    def data_dependencies(self) -> set[str]:
        return strain_data_dependencies(self)

    def introduction_age_mask(
        self,
        age_bins: list[AgeBin] | tuple[AgeBin, ...],
    ) -> list[int]:
        """Convert introduction_ages into a mask over model age bins."""
        return build_introduction_age_mask(strain=self, age_bins=age_bins)

    def evaluated_values(
        self,
        context: dict[str, Any] | None = None,
        data: Any | None = None,
    ) -> dict[str, Any]:
        """Evaluate strain-level values against a runtime context."""
        return evaluated_strain_values(
            strain=self,
            context=context,
            data=data,
        )

    def _parameter_values(self) -> tuple[ParameterValue, ...]:
        return strain_parameter_values(self)


StrainSpec.model_rebuild(
    _types_namespace={
        "ParameterValue": ParameterValue,
    }
)
