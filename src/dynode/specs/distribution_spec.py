from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
from typing import Annotated, Any, Literal

import numpy as np
import numpyro.distributions as dist
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from .value_spec import (
    ConstantValueSpec,
    DistributionValue,
    coerce_value_fields,
)


class DistributionSpec(BaseModel, ABC):
    """
    Base class for declarative NumPyro distribution specs.

    Subclasses should be serializable and should not store raw NumPyro
    Distribution objects.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )

    type: str

    @abstractmethod
    def dependencies(self) -> set[str]:
        """
        Return parameter names needed to construct this distribution.
        """
        raise NotImplementedError

    @abstractmethod
    def to_numpyro(
        self,
        context: dict[str, Any] | None = None,
        data: Any | None = None,
    ) -> dist.Distribution:
        """
        Compile this spec into a NumPyro Distribution.
        """
        raise NotImplementedError


def _contains_bool_or_date(value: Any) -> bool:
    if isinstance(value, (bool, date)):
        return True

    if isinstance(value, list):
        return any(_contains_bool_or_date(item) for item in value)

    return False


def _constant_as_numeric_array(
    value: ConstantValueSpec,
    field_name: str,
) -> np.ndarray:
    """
    Convert a ConstantValueSpec payload into a numeric NumPy array for validation.

    Distribution parameters should be numeric, not bool/date.
    """
    raw = value.value

    if _contains_bool_or_date(raw):
        raise ValueError(f"{field_name} must be numeric, not bool/date.")

    try:
        return np.asarray(raw, dtype=float)
    except Exception as exc:
        raise ValueError(f"{field_name} must be numeric.") from exc


def _validate_constant_numeric(
    value: DistributionValue,
    field_name: str,
) -> None:
    """
    Validate that a constant distribution parameter is numeric.

    ParamRef / DeterministicRef / expressions are checked at runtime.
    """
    if not isinstance(value, ConstantValueSpec):
        return

    _constant_as_numeric_array(value, field_name)


def _validate_constant_positive(
    value: DistributionValue,
    field_name: str,
) -> None:
    """
    Validate that a constant distribution parameter is strictly positive.

    ParamRef / DeterministicRef / expressions are checked at runtime.
    """
    if not isinstance(value, ConstantValueSpec):
        return

    arr = _constant_as_numeric_array(value, field_name)

    if np.any(arr <= 0):
        raise ValueError(f"{field_name} must be positive.")


def _validate_constant_bounds(
    low: DistributionValue | None,
    high: DistributionValue | None,
) -> None:
    """
    Validate high > low when both are constant values.
    """
    if low is None or high is None:
        return

    if not isinstance(low, ConstantValueSpec):
        return

    if not isinstance(high, ConstantValueSpec):
        return

    low_arr = _constant_as_numeric_array(low, "TruncatedNormal low")
    high_arr = _constant_as_numeric_array(high, "TruncatedNormal high")

    try:
        invalid = np.any(high_arr <= low_arr)
    except ValueError as exc:
        raise ValueError(
            "TruncatedNormal low and high constants are not broadcast-compatible."
        ) from exc

    if invalid:
        raise ValueError("TruncatedNormal high must be greater than low.")


class NormalSpec(DistributionSpec):
    type: Literal["normal"] = "normal"

    loc: DistributionValue = Field(
        default_factory=lambda: ConstantValueSpec(value=0.0)
    )
    scale: DistributionValue = Field(
        default_factory=lambda: ConstantValueSpec(value=1.0)
    )

    @model_validator(mode="before")
    @classmethod
    def coerce_values(cls, data: Any) -> Any:
        return coerce_value_fields(data, ("loc", "scale"))

    @model_validator(mode="after")
    def validate_parameters(self) -> Self:
        _validate_constant_numeric(self.loc, "Normal loc")
        _validate_constant_positive(self.scale, "Normal scale")
        return self

    def dependencies(self) -> set[str]:
        return self.loc.dependencies() | self.scale.dependencies()

    def to_numpyro(
        self,
        context: dict[str, Any] | None = None,
        data: Any | None = None,
    ) -> dist.Distribution:
        return dist.Normal(
            loc=self.loc.evaluate(context=context, data=data),
            scale=self.scale.evaluate(context=context, data=data),
        )


class LogNormalSpec(DistributionSpec):
    type: Literal["lognormal"] = "lognormal"

    loc: DistributionValue = Field(
        default_factory=lambda: ConstantValueSpec(value=0.0)
    )
    scale: DistributionValue = Field(
        default_factory=lambda: ConstantValueSpec(value=1.0)
    )

    @model_validator(mode="before")
    @classmethod
    def coerce_values(cls, data: Any) -> Any:
        return coerce_value_fields(data, ("loc", "scale"))

    @model_validator(mode="after")
    def validate_parameters(self) -> Self:
        _validate_constant_numeric(self.loc, "LogNormal loc")
        _validate_constant_positive(self.scale, "LogNormal scale")
        return self

    def dependencies(self) -> set[str]:
        return self.loc.dependencies() | self.scale.dependencies()

    def to_numpyro(
        self,
        context: dict[str, Any] | None = None,
        data: Any | None = None,
    ) -> dist.Distribution:
        return dist.LogNormal(
            loc=self.loc.evaluate(context=context, data=data),
            scale=self.scale.evaluate(context=context, data=data),
        )


class GammaSpec(DistributionSpec):
    type: Literal["gamma"] = "gamma"

    concentration: DistributionValue
    rate: DistributionValue

    @model_validator(mode="before")
    @classmethod
    def coerce_values(cls, data: Any) -> Any:
        return coerce_value_fields(data, ("concentration", "rate"))

    @model_validator(mode="after")
    def validate_parameters(self) -> Self:
        _validate_constant_positive(
            self.concentration,
            "Gamma concentration",
        )
        _validate_constant_positive(
            self.rate,
            "Gamma rate",
        )
        return self

    def dependencies(self) -> set[str]:
        return self.concentration.dependencies() | self.rate.dependencies()

    def to_numpyro(
        self,
        context: dict[str, Any] | None = None,
        data: Any | None = None,
    ) -> dist.Distribution:
        return dist.Gamma(
            concentration=self.concentration.evaluate(
                context=context, data=data
            ),
            rate=self.rate.evaluate(context=context, data=data),
        )


class ExponentialSpec(DistributionSpec):
    type: Literal["exponential"] = "exponential"

    rate: DistributionValue

    @model_validator(mode="before")
    @classmethod
    def coerce_values(cls, data: Any) -> Any:
        return coerce_value_fields(data, ("rate",))

    @model_validator(mode="after")
    def validate_parameters(self) -> Self:
        _validate_constant_positive(
            self.rate,
            "Exponential rate",
        )
        return self

    def dependencies(self) -> set[str]:
        return self.rate.dependencies()

    def to_numpyro(
        self,
        context: dict[str, Any] | None = None,
        data: Any | None = None,
    ) -> dist.Distribution:
        return dist.Exponential(
            rate=self.rate.evaluate(context=context, data=data),
        )


class BetaSpec(DistributionSpec):
    type: Literal["beta"] = "beta"

    concentration1: DistributionValue
    concentration0: DistributionValue

    @model_validator(mode="before")
    @classmethod
    def coerce_values(cls, data: Any) -> Any:
        return coerce_value_fields(
            data,
            ("concentration1", "concentration0"),
        )

    @model_validator(mode="after")
    def validate_parameters(self) -> Self:
        _validate_constant_positive(
            self.concentration1,
            "Beta concentration1",
        )
        _validate_constant_positive(
            self.concentration0,
            "Beta concentration0",
        )
        return self

    def dependencies(self) -> set[str]:
        return (
            self.concentration1.dependencies()
            | self.concentration0.dependencies()
        )

    def to_numpyro(
        self,
        context: dict[str, Any] | None = None,
        data: Any | None = None,
    ) -> dist.Distribution:
        return dist.Beta(
            concentration1=self.concentration1.evaluate(
                context=context, data=data
            ),
            concentration0=self.concentration0.evaluate(
                context=context, data=data
            ),
        )


class HalfNormalSpec(DistributionSpec):
    type: Literal["half_normal"] = "half_normal"

    scale: DistributionValue

    @model_validator(mode="before")
    @classmethod
    def coerce_values(cls, data: Any) -> Any:
        return coerce_value_fields(data, ("scale",))

    @model_validator(mode="after")
    def validate_parameters(self) -> Self:
        _validate_constant_positive(
            self.scale,
            "HalfNormal scale",
        )
        return self

    def dependencies(self) -> set[str]:
        return self.scale.dependencies()

    def to_numpyro(
        self,
        context: dict[str, Any] | None = None,
        data: Any | None = None,
    ) -> dist.Distribution:
        return dist.HalfNormal(
            scale=self.scale.evaluate(context=context, data=data),
        )


class TruncatedNormalSpec(DistributionSpec):
    type: Literal["truncated_normal"] = "truncated_normal"

    loc: DistributionValue = Field(
        default_factory=lambda: ConstantValueSpec(value=0.0)
    )
    scale: DistributionValue = Field(
        default_factory=lambda: ConstantValueSpec(value=1.0)
    )
    low: DistributionValue | None = None
    high: DistributionValue | None = None

    @model_validator(mode="before")
    @classmethod
    def coerce_values(cls, data: Any) -> Any:
        return coerce_value_fields(
            data,
            ("loc", "scale", "low", "high"),
        )

    @model_validator(mode="after")
    def validate_parameters(self) -> Self:
        _validate_constant_numeric(
            self.loc,
            "TruncatedNormal loc",
        )
        _validate_constant_positive(
            self.scale,
            "TruncatedNormal scale",
        )

        if self.low is not None:
            _validate_constant_numeric(
                self.low,
                "TruncatedNormal low",
            )

        if self.high is not None:
            _validate_constant_numeric(
                self.high,
                "TruncatedNormal high",
            )

        _validate_constant_bounds(
            self.low,
            self.high,
        )

        return self

    def dependencies(self) -> set[str]:
        deps = self.loc.dependencies() | self.scale.dependencies()

        if self.low is not None:
            deps |= self.low.dependencies()

        if self.high is not None:
            deps |= self.high.dependencies()

        return deps

    def to_numpyro(
        self,
        context: dict[str, Any] | None = None,
        data: Any | None = None,
    ) -> dist.Distribution:
        return dist.TruncatedNormal(
            loc=self.loc.evaluate(context=context, data=data),
            scale=self.scale.evaluate(context=context, data=data),
            low=None
            if self.low is None
            else self.low.evaluate(context=context, data=data),
            high=None
            if self.high is None
            else self.high.evaluate(context=context, data=data),
        )


class UniformSpec(DistributionSpec):
    type: Literal["uniform"] = "uniform"

    low: DistributionValue = Field(
        default_factory=lambda: ConstantValueSpec(value=0.0)
    )
    high: DistributionValue = Field(
        default_factory=lambda: ConstantValueSpec(value=1.0)
    )

    @model_validator(mode="before")
    @classmethod
    def coerce_values(cls, data: Any) -> Any:
        return coerce_value_fields(data, ("low", "high"))

    @model_validator(mode="after")
    def validate_parameters(self) -> Self:
        _validate_constant_numeric(self.low, "Uniform low")
        _validate_constant_numeric(self.high, "Uniform high")
        _validate_constant_bounds(self.low, self.high)
        return self

    def dependencies(self) -> set[str]:
        return self.low.dependencies() | self.high.dependencies()

    def to_numpyro(
        self, context: dict[str, Any] | None = None, data: Any | None = None
    ) -> dist.Distribution:
        return dist.Uniform(
            low=self.low.evaluate(context=context, data=data),
            high=self.high.evaluate(context=context, data=data),
        )


class AffineTransformSpec(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, arbitrary_types_allowed=True
    )

    type: Literal["affine"] = "affine"
    loc: DistributionValue = Field(
        default_factory=lambda: ConstantValueSpec(value=0.0)
    )
    scale: DistributionValue = Field(
        default_factory=lambda: ConstantValueSpec(value=1.0)
    )

    @model_validator(mode="before")
    @classmethod
    def coerce_values(cls, data: Any) -> Any:
        return coerce_value_fields(data, ("loc", "scale"))

    def dependencies(self) -> set[str]:
        return self.loc.dependencies() | self.scale.dependencies()

    def to_numpyro(
        self, context: dict[str, Any] | None = None, data: Any | None = None
    ):
        import numpyro.distributions.transforms as transforms

        return transforms.AffineTransform(
            loc=self.loc.evaluate(context=context, data=data),
            scale=self.scale.evaluate(context=context, data=data),
        )


DistributionTransformSpec = Annotated[
    AffineTransformSpec,
    Field(discriminator="type"),
]


class TransformedDistributionSpec(DistributionSpec):
    type: Literal["transformed"] = "transformed"

    base: PriorDistributionSpec
    transforms: tuple[DistributionTransformSpec, ...]

    def dependencies(self) -> set[str]:
        deps = self.base.dependencies()
        for transform in self.transforms:
            deps |= transform.dependencies()
        return deps

    def to_numpyro(
        self, context: dict[str, Any] | None = None, data: Any | None = None
    ) -> dist.Distribution:
        return dist.TransformedDistribution(
            self.base.to_numpyro(context=context, data=data),
            [
                transform.to_numpyro(context=context, data=data)
                for transform in self.transforms
            ],
        )


class RegisteredDistributionSpec(DistributionSpec):
    type: Literal["registered"] = "registered"

    registry_key: str
    kwargs: dict[str, DistributionValue] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def coerce_kwargs(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        kwargs = data.get("kwargs")
        if isinstance(kwargs, dict):
            data["kwargs"] = {
                key: value
                if isinstance(value, dict) and "type" in value
                else {"type": "constant", "value": value}
                for key, value in kwargs.items()
            }
        return data

    def dependencies(self) -> set[str]:
        deps: set[str] = set()
        for value in self.kwargs.values():
            deps |= value.dependencies()
        return deps

    def to_numpyro(
        self, context: dict[str, Any] | None = None, data: Any | None = None
    ) -> dist.Distribution:
        registry = None
        if isinstance(data, dict):
            registry = data.get("distribution_registry")
        if registry is None:
            registry = (context or {}).get("distribution_registry")
        if registry is None or self.registry_key not in registry:
            raise KeyError(
                f"Unknown registered distribution {self.registry_key!r}."
            )
        kwargs = {
            key: value.evaluate(context=context, data=data)
            for key, value in self.kwargs.items()
        }
        return registry[self.registry_key](**kwargs)


PriorDistributionSpec = Annotated[
    NormalSpec
    | LogNormalSpec
    | GammaSpec
    | ExponentialSpec
    | BetaSpec
    | HalfNormalSpec
    | TruncatedNormalSpec
    | UniformSpec
    | TransformedDistributionSpec
    | RegisteredDistributionSpec,
    Field(discriminator="type"),
]

# Rebuild models that contain forward references to PriorDistributionSpec.
try:
    TransformedDistributionSpec.model_rebuild(
        _types_namespace={"PriorDistributionSpec": PriorDistributionSpec}
    )
except Exception:
    pass
