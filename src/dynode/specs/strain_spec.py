from __future__ import annotations

from datetime import date
from typing import Annotated, Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from dynode.typing import DynodeName

from . import InteractionSpec
from .bin_spec import AgeBin
from .value_spec import (
    ConstantValueSpec,
    ParameterValue,
    coerce_value_fields,
)


DoseCount = Annotated[int, Field(ge=0)]
Probability = Annotated[float, Field(ge=0.0, le=1.0)]


class StrainSpec(BaseModel):
    """
    Declarative strain specification.

    This class describes strain-level parameters and validation rules.

    It should not:
    - sample NumPyro distributions
    - mutate runtime fields
    - build JAX arrays directly
    - store raw NumPyro Distribution objects

    Strain-level parameter fields use the shared value_spec.py system, so they
    may be constants, parameter references, deterministic references, or
    expressions.
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
        """
        Allow simple Python/YAML forms like:

            r0: 2.5

        instead of requiring:

            r0:
              type: constant
              value: 2.5
        """
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
        self._validate_core_values()
        self._validate_introduction_fields()
        self._validate_no_data_dependencies()

        return self

    @property
    def parameter_dependencies(self) -> set[str]:
        deps: set[str] = set()

        for value in self._parameter_values():
            deps |= value.parameter_dependencies()

        for interaction in self.interactions.values():
            deps |= self._dependency_set(interaction, "parameter_dependencies")

        return deps

    @property
    def deterministic_dependencies(self) -> set[str]:
        deps: set[str] = set()

        for value in self._parameter_values():
            deps |= value.deterministic_dependencies()

        for interaction in self.interactions.values():
            deps |= self._dependency_set(interaction, "deterministic_dependencies")

        return deps

    @property
    def dependencies(self) -> set[str]:
        """
        All parameter-like dependencies needed to evaluate this strain.

        Includes sampled parameter refs and deterministic refs.
        """
        return self.parameter_dependencies | self.deterministic_dependencies

    @property
    def data_dependencies(self) -> set[str]:
        deps: set[str] = set()

        for value in self._parameter_values():
            deps |= value.data_dependencies()

        for interaction in self.interactions.values():
            deps |= self._dependency_set(interaction, "data_dependencies")

        return deps

    def introduction_age_mask(self, age_bins: list[AgeBin] | tuple[AgeBin, ...]) -> list[int]:
        """
        Convert introduction_ages into a mask over model age bins.

        This replaces the old mutable introduction_ages_mask_vector field.
        """
        if not self.is_introduced or self.introduction_ages is None:
            return [0 for _ in age_bins]

        missing = [
            age
            for age in self.introduction_ages
            if age not in age_bins
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

    def evaluated_values(
        self,
        context: dict[str, Any] | None = None,
        data: Any | None = None,
    ) -> dict[str, Any]:
        """
        Evaluate strain-level values against a runtime context.

        This is useful in the runtime/compiler layer after priors and
        deterministic parameters have been resolved.
        """
        values: dict[str, Any] = {
            "name": self.name,
            "r0": self.r0.evaluate(context=context, data=data),
            "infectious_period": self.infectious_period.evaluate(
                context=context,
                data=data,
            ),
            "is_introduced": self.is_introduced,
        }

        if self.exposed_to_infectious is not None:
            values["exposed_to_infectious"] = self.exposed_to_infectious.evaluate(
                context=context,
                data=data,
            )

        if self.vaccine_efficacy is not None:
            values["vaccine_efficacy"] = dict(self.vaccine_efficacy)

        if self.is_introduced:
            values["introduction_time"] = self.introduction_time.evaluate(
                context=context,
                data=data,
            )
            values["introduction_percentage"] = self.introduction_percentage.evaluate(
                context=context,
                data=data,
            )
            values["introduction_scale"] = self.introduction_scale.evaluate(
                context=context,
                data=data,
            )
            values["introduction_ages"] = self.introduction_ages

        return values

    def _validate_core_values(self) -> None:
        self._validate_constant_non_negative(
            self.r0,
            field_name="r0",
        )

        self._validate_constant_positive(
            self.infectious_period,
            field_name="infectious_period",
        )

        if self.exposed_to_infectious is not None:
            self._validate_constant_positive(
                self.exposed_to_infectious,
                field_name="exposed_to_infectious",
            )

    def _validate_introduction_fields(self) -> None:
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

            return

        required = (
            "introduction_time",
            "introduction_percentage",
            "introduction_scale",
        )

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

        self._validate_constant_non_negative_or_date(
            self.introduction_time,
            field_name="introduction_time",
        )

        self._validate_constant_positive(
            self.introduction_percentage,
            field_name="introduction_percentage",
        )

        self._validate_constant_positive(
            self.introduction_scale,
            field_name="introduction_scale",
        )

    def _validate_no_data_dependencies(self) -> None:
        """
        Strain parameter specs should generally not depend on observed data.

        Initializers may depend on data, but strain dynamics should be
        parameter-driven. Relax this if your framework intentionally supports
        data-derived strain metadata.
        """
        data_deps = self.data_dependencies

        if data_deps:
            raise ValueError(
                f"Strain {self.name!r} contains data references, which are not "
                f"allowed in StrainSpec. Data dependencies: {sorted(data_deps)}."
            )

    def _parameter_values(self) -> tuple[ParameterValue, ...]:
        values: list[ParameterValue] = [
            self.r0,
            self.infectious_period,
        ]

        optional_values = (
            self.exposed_to_infectious,
            self.introduction_time,
            self.introduction_percentage,
            self.introduction_scale,
        )

        for value in optional_values:
            if value is not None:
                values.append(value)

        return tuple(values)

    @staticmethod
    def _dependency_set(obj: Any, attr_name: str) -> set[str]:
        attr = getattr(obj, attr_name, None)

        if attr is None:
            return set()

        if callable(attr):
            return set(attr())

        return set(attr)

    @classmethod
    def _validate_constant_non_negative(
        cls,
        value: ParameterValue,
        field_name: str,
    ) -> None:
        if not isinstance(value, ConstantValueSpec):
            return

        array = cls._constant_numeric_array(value, field_name)

        if np.any(array < 0):
            raise ValueError(f"{field_name} must be non-negative.")

    @classmethod
    def _validate_constant_positive(
        cls,
        value: ParameterValue,
        field_name: str,
    ) -> None:
        if not isinstance(value, ConstantValueSpec):
            return

        array = cls._constant_numeric_array(value, field_name)

        if np.any(array <= 0):
            raise ValueError(f"{field_name} must be positive.")

    @classmethod
    def _validate_constant_non_negative_or_date(
        cls,
        value: ParameterValue,
        field_name: str,
    ) -> None:
        if not isinstance(value, ConstantValueSpec):
            return

        raw = value.value

        if isinstance(raw, date):
            return

        array = cls._constant_numeric_array(value, field_name)

        if np.any(array < 0):
            raise ValueError(
                f"{field_name} must be a date or non-negative simulation time."
            )

    @staticmethod
    def _constant_numeric_array(
        value: ConstantValueSpec,
        field_name: str,
    ) -> np.ndarray:
        raw = value.value

        if isinstance(raw, bool):
            raise ValueError(f"{field_name} must be numeric, not bool.")

        if isinstance(raw, date):
            raise ValueError(f"{field_name} must be numeric, not date.")

        try:
            return np.asarray(raw, dtype=float)
        except Exception as exc:
            raise ValueError(f"{field_name} must be numeric.") from exc
