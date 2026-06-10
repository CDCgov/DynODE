from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Annotated, Any, Literal

import jax.numpy as jnp
import numpyro.distributions as dist
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self


class DistributionValueSpec(BaseModel):
    """
    Base class for values used inside distribution parameters.

    Examples
    --------
    loc=ConstantValueSpec(value=0.0)
    scale=ParamRef(name="sigma_scale")
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )

    type: str

    def dependencies(self) -> set[str]:
        raise NotImplementedError

    def evaluate(self, context: dict[str, Any] | None = None) -> Any:
        raise NotImplementedError


class ConstantValueSpec(DistributionValueSpec):
    type: Literal["constant"] = "constant"

    value: int | float | bool | list[int] | list[float]

    def dependencies(self) -> set[str]:
        return set()

    def evaluate(self, context: dict[str, Any] | None = None) -> Any:
        if isinstance(self.value, list):
            return jnp.asarray(self.value)

        return self.value


class ParamRef(DistributionValueSpec):
    """
    Reference to a sampled or resolved parameter.
    """

    type: Literal["param_ref"] = "param_ref"

    name: str

    def dependencies(self) -> set[str]:
        return {self.name}

    def evaluate(self, context: dict[str, Any] | None = None) -> Any:
        if context is None:
            raise ValueError(
                f"Cannot resolve parameter reference {self.name!r} without context."
            )

        try:
            return context[self.name]
        except KeyError as exc:
            raise KeyError(
                f"Parameter {self.name!r} was not found in context. "
                f"Available values are: {sorted(context)}."
            ) from exc


class DeterministicRef(DistributionValueSpec):
    """
    Reference to a deterministic parameter.
    """

    type: Literal["deterministic_ref"] = "deterministic_ref"

    name: str

    def dependencies(self) -> set[str]:
        return {self.name}

    def evaluate(self, context: dict[str, Any] | None = None) -> Any:
        if context is None:
            raise ValueError(
                f"Cannot resolve deterministic reference {self.name!r} without context."
            )

        try:
            return context[self.name]
        except KeyError as exc:
            raise KeyError(
                f"Deterministic parameter {self.name!r} was not found in context. "
                f"Available values are: {sorted(context)}."
            ) from exc


DistributionValue = Annotated[
    ConstantValueSpec | ParamRef | DeterministicRef,
    Field(discriminator="type"),
]


def as_value(value: Any) -> DistributionValue:
    """
    Convenience helper for Python-authored specs.

    This lets you write:

        NormalSpec(loc=0.0, scale=1.0)

    instead of:

        NormalSpec(
            loc=ConstantValueSpec(value=0.0),
            scale=ConstantValueSpec(value=1.0),
        )
    """
    if isinstance(value, DistributionValueSpec):
        return value

    return ConstantValueSpec(value=value)


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
    ) -> dist.Distribution:
        """
        Compile this spec into a NumPyro Distribution.
        """
        raise NotImplementedError

    def _eval(
        self,
        value: DistributionValue,
        context: dict[str, Any] | None,
    ) -> Any:
        return value.evaluate(context)

class NormalSpec(DistributionSpec):
    type: Literal["normal"] = "normal"

    loc: DistributionValue = Field(default_factory=lambda: ConstantValueSpec(value=0.0))
    scale: DistributionValue = Field(default_factory=lambda: ConstantValueSpec(value=1.0))

    @model_validator(mode="before")
    @classmethod
    def coerce_values(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)
            if "loc" in data:
                data["loc"] = as_value(data["loc"])
            if "scale" in data:
                data["scale"] = as_value(data["scale"])
        return data

    @model_validator(mode="after")
    def validate_constant_scale(self) -> Self:
        if isinstance(self.scale, ConstantValueSpec):
            scale = self.scale.value
            if isinstance(scale, (int, float)) and scale <= 0:
                raise ValueError("Normal scale must be positive.")
        return self

    def dependencies(self) -> set[str]:
        return self.loc.dependencies() | self.scale.dependencies()

    def to_numpyro(self, context: dict[str, Any] | None = None) -> dist.Distribution:
        return dist.Normal(
            loc=self._eval(self.loc, context),
            scale=self._eval(self.scale, context),
        )


class LogNormalSpec(DistributionSpec):
    type: Literal["lognormal"] = "lognormal"

    loc: DistributionValue = Field(default_factory=lambda: ConstantValueSpec(value=0.0))
    scale: DistributionValue = Field(default_factory=lambda: ConstantValueSpec(value=1.0))

    @model_validator(mode="before")
    @classmethod
    def coerce_values(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)
            if "loc" in data:
                data["loc"] = as_value(data["loc"])
            if "scale" in data:
                data["scale"] = as_value(data["scale"])
        return data

    @model_validator(mode="after")
    def validate_constant_scale(self) -> Self:
        if isinstance(self.scale, ConstantValueSpec):
            scale = self.scale.value
            if isinstance(scale, (int, float)) and scale <= 0:
                raise ValueError("LogNormal scale must be positive.")
        return self

    def dependencies(self) -> set[str]:
        return self.loc.dependencies() | self.scale.dependencies()

    def to_numpyro(self, context: dict[str, Any] | None = None) -> dist.Distribution:
        return dist.LogNormal(
            loc=self._eval(self.loc, context),
            scale=self._eval(self.scale, context),
        )


class GammaSpec(DistributionSpec):
    type: Literal["gamma"] = "gamma"

    concentration: DistributionValue
    rate: DistributionValue

    @model_validator(mode="before")
    @classmethod
    def coerce_values(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)
            if "concentration" in data:
                data["concentration"] = as_value(data["concentration"])
            if "rate" in data:
                data["rate"] = as_value(data["rate"])
        return data

    @model_validator(mode="after")
    def validate_constant_values(self) -> Self:
        for field_name in ("concentration", "rate"):
            value = getattr(self, field_name)

            if isinstance(value, ConstantValueSpec):
                raw = value.value
                if isinstance(raw, (int, float)) and raw <= 0:
                    raise ValueError(f"Gamma {field_name} must be positive.")

        return self

    def dependencies(self) -> set[str]:
        return self.concentration.dependencies() | self.rate.dependencies()

    def to_numpyro(self, context: dict[str, Any] | None = None) -> dist.Distribution:
        return dist.Gamma(
            concentration=self._eval(self.concentration, context),
            rate=self._eval(self.rate, context),
        )


class ExponentialSpec(DistributionSpec):
    type: Literal["exponential"] = "exponential"

    rate: DistributionValue

    @model_validator(mode="before")
    @classmethod
    def coerce_values(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)
            if "rate" in data:
                data["rate"] = as_value(data["rate"])
        return data

    @model_validator(mode="after")
    def validate_constant_rate(self) -> Self:
        if isinstance(self.rate, ConstantValueSpec):
            rate = self.rate.value
            if isinstance(rate, (int, float)) and rate <= 0:
                raise ValueError("Exponential rate must be positive.")
        return self

    def dependencies(self) -> set[str]:
        return self.rate.dependencies()

    def to_numpyro(self, context: dict[str, Any] | None = None) -> dist.Distribution:
        return dist.Exponential(
            rate=self._eval(self.rate, context),
        )


class BetaSpec(DistributionSpec):
    type: Literal["beta"] = "beta"

    concentration1: DistributionValue
    concentration0: DistributionValue

    @model_validator(mode="before")
    @classmethod
    def coerce_values(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)
            if "concentration1" in data:
                data["concentration1"] = as_value(data["concentration1"])
            if "concentration0" in data:
                data["concentration0"] = as_value(data["concentration0"])
        return data

    @model_validator(mode="after")
    def validate_constant_values(self) -> Self:
        for field_name in ("concentration1", "concentration0"):
            value = getattr(self, field_name)

            if isinstance(value, ConstantValueSpec):
                raw = value.value
                if isinstance(raw, (int, float)) and raw <= 0:
                    raise ValueError(f"Beta {field_name} must be positive.")

        return self

    def dependencies(self) -> set[str]:
        return (
            self.concentration1.dependencies()
            | self.concentration0.dependencies()
        )

    def to_numpyro(self, context: dict[str, Any] | None = None) -> dist.Distribution:
        return dist.Beta(
            concentration1=self._eval(self.concentration1, context),
            concentration0=self._eval(self.concentration0, context),
        )


class HalfNormalSpec(DistributionSpec):
    type: Literal["half_normal"] = "half_normal"

    scale: DistributionValue

    @model_validator(mode="before")
    @classmethod
    def coerce_values(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)
            if "scale" in data:
                data["scale"] = as_value(data["scale"])
        return data

    @model_validator(mode="after")
    def validate_constant_scale(self) -> Self:
        if isinstance(self.scale, ConstantValueSpec):
            scale = self.scale.value
            if isinstance(scale, (int, float)) and scale <= 0:
                raise ValueError("HalfNormal scale must be positive.")
        return self

    def dependencies(self) -> set[str]:
        return self.scale.dependencies()

    def to_numpyro(self, context: dict[str, Any] | None = None) -> dist.Distribution:
        return dist.HalfNormal(
            scale=self._eval(self.scale, context),
        )


class TruncatedNormalSpec(DistributionSpec):
    type: Literal["truncated_normal"] = "truncated_normal"

    loc: DistributionValue = Field(default_factory=lambda: ConstantValueSpec(value=0.0))
    scale: DistributionValue = Field(default_factory=lambda: ConstantValueSpec(value=1.0))
    low: DistributionValue | None = None
    high: DistributionValue | None = None

    @model_validator(mode="before")
    @classmethod
    def coerce_values(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)

            for key in ("loc", "scale", "low", "high"):
                if key in data and data[key] is not None:
                    data[key] = as_value(data[key])

        return data

    @model_validator(mode="after")
    def validate_constant_values(self) -> Self:
        if isinstance(self.scale, ConstantValueSpec):
            scale = self.scale.value
            if isinstance(scale, (int, float)) and scale <= 0:
                raise ValueError("TruncatedNormal scale must be positive.")

        if (
            isinstance(self.low, ConstantValueSpec)
            and isinstance(self.high, ConstantValueSpec)
            and isinstance(self.low.value, (int, float))
            and isinstance(self.high.value, (int, float))
            and self.high.value <= self.low.value
        ):
            raise ValueError("TruncatedNormal high must be greater than low.")

        return self

    def dependencies(self) -> set[str]:
        deps = self.loc.dependencies() | self.scale.dependencies()

        if self.low is not None:
            deps |= self.low.dependencies()

        if self.high is not None:
            deps |= self.high.dependencies()

        return deps

    def to_numpyro(self, context: dict[str, Any] | None = None) -> dist.Distribution:
        return dist.TruncatedNormal(
            loc=self._eval(self.loc, context),
            scale=self._eval(self.scale, context),
            low=None if self.low is None else self._eval(self.low, context),
            high=None if self.high is None else self._eval(self.high, context),
        )

PriorDistributionSpec = Annotated[
    NormalSpec
    | LogNormalSpec
    | GammaSpec
    | ExponentialSpec
    | BetaSpec
    | HalfNormalSpec
    | TruncatedNormalSpec,
    Field(discriminator="type"),
]
