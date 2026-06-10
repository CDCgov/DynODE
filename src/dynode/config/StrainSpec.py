from datetime import date
from typing import Annotated, Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    PositiveFloat,
    model_validator,
)
from typing_extensions import Self

from dynode.typing import DynodeName

from . import InteractionSpec
from .bins import AgeBin


class ParamRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str


class DeterministicRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str


ParameterValue = float | int | date | ParamRef | DeterministicRef


class StrainSpec(BaseModel):
    """
    Declarative strain specification.

    This class describes strain-level parameters and validation rules.
    It should not sample NumPyro distributions or mutate runtime fields.
    """

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
        validate_assignment=True,
    )

    name: DynodeName = Field(
        description="Strain name, no leading numbers or special characters."
    )

    r0: ParameterValue = Field(
        description="Strain reproduction number used to calculate transmission rate."
    )

    infectious_period: ParameterValue = Field(
        description="Average number of days a freshly infectious population stays infectious."
    )

    exposed_to_infectious: ParameterValue | None = Field(
        default=None,
        description=(
            "Average number of days between exposure to this strain and becoming infectious."
        ),
    )

    vaccine_efficacy: dict[int, NonNegativeFloat] | None = Field(
        default=None,
        description=(
            "Maps vaccine dose count to protection against infection before immune waning."
        ),
    )

    interactions: dict[str, InteractionSpec] = Field(
        default_factory=dict,
        description=(
            "Explicit source-strain to target-strain interaction specs. "
            "Keys are target strain names."
        ),
    )

    is_introduced: bool = Field(
        default=False,
        description=(
            "Whether this strain is introduced from an external population during simulation."
        ),
    )

    introduction_time: ParameterValue | None = Field(
        default=None,
        description=(
            "Date or simulation time of peak external infectious population mixing. "
            "Only used if is_introduced is True."
        ),
    )

    introduction_percentage: ParameterValue | None = Field(
        default=None,
        description=(
            "External population size relative to tracked population. "
            "Only used if is_introduced is True."
        ),
    )

    introduction_scale: ParameterValue | None = Field(
        default=None,
        description=(
            "Spread of external infectious population mixing around introduction_time. "
            "Only used if is_introduced is True."
        ),
    )

    introduction_ages: list[AgeBin] | None = Field(
        default=None,
        description=(
            "Age bins receiving external introductions. "
            "Only used if is_introduced is True."
        ),
    )

    @model_validator(mode="after")
    def validate_core_values(self) -> Self:
        if isinstance(self.r0, (int, float)) and self.r0 < 0:
            raise ValueError("r0 must be non-negative.")

        if (
            isinstance(self.infectious_period, (int, float))
            and self.infectious_period <= 0
        ):
            raise ValueError("infectious_period must be positive.")

        if (
            self.exposed_to_infectious is not None
            and isinstance(self.exposed_to_infectious, (int, float))
            and self.exposed_to_infectious <= 0
        ):
            raise ValueError(
                "exposed_to_infectious must be positive when provided."
            )

        return self

    @model_validator(mode="after")
    def validate_vaccine_efficacy(self) -> Self:
        if self.vaccine_efficacy is None:
            return self

        for dose_count, efficacy in self.vaccine_efficacy.items():
            if dose_count < 0:
                raise ValueError(
                    "vaccine_efficacy dose counts must be non-negative integers."
                )

            if efficacy < 0:
                raise ValueError(
                    "vaccine_efficacy values must be non-negative."
                )

            if efficacy > 1:
                raise ValueError("vaccine_efficacy values should be <= 1.0.")

        return self

    @model_validator(mode="after")
    def validate_introduction_fields(self) -> Self:
        introduction_fields = {
            "introduction_time": self.introduction_time,
            "introduction_percentage": self.introduction_percentage,
            "introduction_scale": self.introduction_scale,
            "introduction_ages": self.introduction_ages,
        }

        provided = [
            field_name
            for field_name, value in introduction_fields.items()
            if value is not None
        ]

        if not self.is_introduced:
            if provided:
                raise ValueError(
                    "Introduction fields were provided, but is_introduced=False. "
                    f"Provided fields: {provided}."
                )
            return self

        required = [
            "introduction_time",
            "introduction_percentage",
            "introduction_scale",
        ]

        missing = [
            field_name
            for field_name in required
            if getattr(self, field_name) is None
        ]

        if missing:
            raise ValueError(
                "Introduced strains must define introduction_time, "
                "introduction_percentage, and introduction_scale. "
                f"Missing: {missing}."
            )

        if (
            isinstance(self.introduction_percentage, (int, float))
            and self.introduction_percentage <= 0
        ):
            raise ValueError("introduction_percentage must be positive.")

        if (
            isinstance(self.introduction_scale, (int, float))
            and self.introduction_scale <= 0
        ):
            raise ValueError("introduction_scale must be positive.")

        return self

    def introduction_age_mask(self, age_bins: list[AgeBin]) -> list[int]:
        """
        Runtime helper.

        Converts introduction_ages into a mask over model age bins.
        This replaces the old mutable introduction_ages_mask_vector field.
        """
        if not self.is_introduced or self.introduction_ages is None:
            return [0 for _ in age_bins]

        missing = [
            age for age in self.introduction_ages if age not in age_bins
        ]

        if missing:
            raise ValueError(
                f"Strain {self.name!r} has introduction_ages not present "
                f"in model age bins: {missing}."
            )

        return [
            1 if age_bin in self.introduction_ages else 0
            for age_bin in age_bins
        ]
