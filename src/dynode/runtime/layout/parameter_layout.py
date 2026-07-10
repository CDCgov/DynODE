from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .utils import duplicates, name_of, readonly_mapping


@dataclass(frozen=True, slots=True)
class RuntimeParameterLayout:
    """
    Compiled parameter layout.

    This stores names and specs needed by parameter_sampling.py. It does not
    call numpyro.sample itself.
    """

    prior_names: tuple[str, ...] = field(default_factory=tuple)
    deterministic_names: tuple[str, ...] = field(default_factory=tuple)
    deterministic_order: tuple[Any, ...] = field(default_factory=tuple)

    prior_specs: Mapping[str, Any] = field(default_factory=dict, repr=False)
    deterministic_specs: Mapping[str, Any] = field(
        default_factory=dict, repr=False
    )

    prior_to_idx: Mapping[str, int] = field(init=False, repr=False)
    deterministic_to_idx: Mapping[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        duplicate_priors = duplicates(self.prior_names)
        if duplicate_priors:
            raise ValueError(f"Duplicate prior names: {duplicate_priors}.")

        duplicate_deterministic = duplicates(self.deterministic_names)
        if duplicate_deterministic:
            raise ValueError(
                f"Duplicate deterministic parameter names: {duplicate_deterministic}."
            )

        collisions = sorted(
            set(self.prior_names) & set(self.deterministic_names)
        )
        if collisions:
            raise ValueError(
                "A parameter cannot be both sampled and deterministic. "
                f"Colliding names: {collisions}."
            )

        ordered_names = tuple(
            name_of(spec) for spec in self.deterministic_order
        )

        if set(ordered_names) != set(self.deterministic_names):
            raise ValueError(
                "deterministic_order must contain exactly the deterministic "
                "parameters listed in deterministic_names. "
                f"deterministic_names={self.deterministic_names}, "
                f"ordered_names={ordered_names}."
            )

        object.__setattr__(
            self,
            "prior_to_idx",
            readonly_mapping(
                {name: i for i, name in enumerate(self.prior_names)}
            ),
        )

        object.__setattr__(
            self,
            "deterministic_to_idx",
            readonly_mapping(
                {name: i for i, name in enumerate(self.deterministic_names)}
            ),
        )

        object.__setattr__(
            self,
            "prior_specs",
            readonly_mapping(self.prior_specs),
        )

        object.__setattr__(
            self,
            "deterministic_specs",
            readonly_mapping(self.deterministic_specs),
        )

    @classmethod
    def from_spec(cls, parameters: Any) -> RuntimeParameterLayout:
        priors = tuple(getattr(parameters, "priors", ()))
        deterministic = tuple(getattr(parameters, "deterministic", ()))

        deterministic_execution_order = getattr(
            parameters,
            "deterministic_execution_order",
            None,
        )

        if callable(deterministic_execution_order):
            deterministic_order = tuple(deterministic_execution_order())
        else:
            deterministic_order = deterministic

        prior_names = tuple(name_of(prior) for prior in priors)
        deterministic_names = tuple(name_of(spec) for spec in deterministic)

        return cls(
            prior_names=prior_names,
            deterministic_names=deterministic_names,
            deterministic_order=deterministic_order,
            prior_specs={name_of(prior): prior for prior in priors},
            deterministic_specs={
                name_of(spec): spec for spec in deterministic
            },
        )

    @property
    def resolved_names(self) -> tuple[str, ...]:
        return self.prior_names + self.deterministic_names

    @property
    def resolved_name_set(self) -> set[str]:
        return set(self.resolved_names)

    def has_parameter(self, name: str) -> bool:
        return name in self.resolved_name_set

    def has_prior(self, name: str) -> bool:
        return name in self.prior_to_idx

    def has_deterministic(self, name: str) -> bool:
        return name in self.deterministic_to_idx

    def get_prior(self, name: str) -> Any:
        try:
            return self.prior_specs[name]
        except KeyError as exc:
            raise KeyError(
                f"Unknown prior {name!r}. Known priors are: {self.prior_names}."
            ) from exc

    def get_deterministic(self, name: str) -> Any:
        try:
            return self.deterministic_specs[name]
        except KeyError as exc:
            raise KeyError(
                f"Unknown deterministic parameter {name!r}. "
                f"Known deterministic parameters are: {self.deterministic_names}."
            ) from exc

    def validate_context(
        self,
        context: Mapping[str, Any],
        *,
        require_all: bool = True,
    ) -> None:
        if not require_all:
            return

        missing = sorted(self.resolved_name_set - set(context))

        if missing:
            raise ValueError(
                f"Parameter context is missing required values: {missing}."
            )
