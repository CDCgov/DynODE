from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import jax.numpy as jnp

from .utils import duplicates, name_of, readonly_mapping


@dataclass(frozen=True, slots=True)
class RuntimeTransmission:
    """
    Compiled transmission layout.

    Stores strain indexes, interaction matrix specs, and introduction age masks.
    """

    strain_names: tuple[str, ...]
    interaction_matrix_spec: tuple[tuple[Any, ...], ...]
    introduction_age_masks: Mapping[str, tuple[int, ...]] = field(
        default_factory=dict
    )
    strain_specs: Mapping[str, Any] = field(default_factory=dict, repr=False)

    strains_to_idx: Mapping[str, int] = field(init=False, repr=False)
    introduced_strain_names: tuple[str, ...] = field(init=False)
    introduced_strain_indices: tuple[int, ...] = field(init=False)
    n_age_bins_for_introduction: int = field(init=False)

    def __post_init__(self) -> None:
        if not self.strain_names:
            raise ValueError(
                "RuntimeTransmission requires at least one strain."
            )

        duplicate_names = duplicates(self.strain_names)
        if duplicate_names:
            raise ValueError(f"Duplicate strain names: {duplicate_names}.")

        n_strains = len(self.strain_names)

        if len(self.interaction_matrix_spec) != n_strains:
            raise ValueError(
                "interaction_matrix_spec must have one row per strain. "
                f"Expected {n_strains}, got {len(self.interaction_matrix_spec)}."
            )

        for row in self.interaction_matrix_spec:
            if len(row) != n_strains:
                raise ValueError(
                    "interaction_matrix_spec must be square with shape "
                    f"({n_strains}, {n_strains})."
                )

        unknown_mask_names = sorted(
            set(self.introduction_age_masks) - set(self.strain_names)
        )

        if unknown_mask_names:
            raise ValueError(
                "introduction_age_masks contains unknown strains: "
                f"{unknown_mask_names}."
            )

        mask_lengths = {
            len(mask) for mask in self.introduction_age_masks.values()
        }

        if len(mask_lengths) > 1:
            raise ValueError(
                "All introduction age masks must have the same length. "
                f"Got lengths: {sorted(mask_lengths)}."
            )

        n_age_bins = next(iter(mask_lengths)) if mask_lengths else 0

        normalized_masks = {
            strain_name: tuple(
                int(value)
                for value in self.introduction_age_masks.get(
                    strain_name,
                    tuple(0 for _ in range(n_age_bins)),
                )
            )
            for strain_name in self.strain_names
        }

        introduced_names: list[str] = []

        for strain_name in self.strain_names:
            strain_spec = self.strain_specs.get(strain_name)
            is_introduced = bool(getattr(strain_spec, "is_introduced", False))

            if is_introduced or any(normalized_masks[strain_name]):
                introduced_names.append(strain_name)

        strains_to_idx = {
            strain_name: i for i, strain_name in enumerate(self.strain_names)
        }

        object.__setattr__(
            self,
            "strains_to_idx",
            readonly_mapping(strains_to_idx),
        )

        object.__setattr__(
            self,
            "introduction_age_masks",
            readonly_mapping(normalized_masks),
        )

        object.__setattr__(
            self,
            "strain_specs",
            readonly_mapping(self.strain_specs),
        )

        object.__setattr__(
            self,
            "introduced_strain_names",
            tuple(introduced_names),
        )

        object.__setattr__(
            self,
            "introduced_strain_indices",
            tuple(strains_to_idx[name] for name in introduced_names),
        )

        object.__setattr__(
            self,
            "n_age_bins_for_introduction",
            n_age_bins,
        )

    @classmethod
    def from_spec(
        cls,
        transmission: Any,
        *,
        age_bins: tuple[Any, ...] = tuple(),
    ) -> RuntimeTransmission:
        strain_names_value = getattr(transmission, "strain_names", None)

        if callable(strain_names_value):
            strain_names_value = strain_names_value()

        if strain_names_value is None:
            strain_names = tuple(
                name_of(strain) for strain in transmission.strains
            )
        else:
            strain_names = tuple(str(name) for name in strain_names_value)

        strains = tuple(getattr(transmission, "strains", ()))

        interaction_matrix = tuple(
            tuple(row) for row in transmission.interaction_matrix_spec()
        )

        masks: dict[str, tuple[int, ...]] = {}

        for strain in strains:
            strain_name = name_of(strain)

            introduction_age_mask = getattr(
                strain, "introduction_age_mask", None
            )

            if callable(introduction_age_mask):
                masks[strain_name] = tuple(
                    int(value) for value in introduction_age_mask(age_bins)
                )
            else:
                masks[strain_name] = tuple(0 for _ in age_bins)

        return cls(
            strain_names=strain_names,
            interaction_matrix_spec=interaction_matrix,
            introduction_age_masks=masks,
            strain_specs={name_of(strain): strain for strain in strains},
        )

    @property
    def n_strains(self) -> int:
        return len(self.strain_names)

    @property
    def has_introductions(self) -> bool:
        return bool(self.introduced_strain_names)

    def has_strain(self, name: str) -> bool:
        return name in self.strains_to_idx

    def index_of(self, strain_name: str) -> int:
        try:
            return self.strains_to_idx[strain_name]
        except KeyError as exc:
            raise KeyError(
                f"Unknown strain {strain_name!r}. Known strains are: {self.strain_names}."
            ) from exc

    def get_strain_spec(self, strain_name: str) -> Any:
        try:
            return self.strain_specs[strain_name]
        except KeyError as exc:
            raise KeyError(
                f"Unknown strain {strain_name!r}. Known strains are: {self.strain_names}."
            ) from exc

    def interaction_spec(
        self,
        source_strain: str,
        target_strain: str,
    ) -> Any:
        source_idx = self.index_of(source_strain)
        target_idx = self.index_of(target_strain)

        return self.interaction_matrix_spec[source_idx][target_idx]

    def interaction_matrix_named(self) -> dict[str, dict[str, Any]]:
        return {
            source_name: {
                target_name: self.interaction_spec(source_name, target_name)
                for target_name in self.strain_names
            }
            for source_name in self.strain_names
        }

    def evaluate_interaction_matrix(
        self,
        context: Mapping[str, Any] | None = None,
        data: Any | None = None,
    ) -> Any:
        rows: list[list[Any]] = []

        for row in self.interaction_matrix_spec:
            evaluated_row: list[Any] = []

            for interaction in row:
                evaluate = getattr(interaction, "evaluate", None)

                if callable(evaluate):
                    evaluated_row.append(
                        evaluate(
                            context=context,
                            data=data,
                        )
                    )
                else:
                    evaluated_row.append(interaction)

            rows.append(evaluated_row)

        return jnp.asarray(rows)

    def introduction_age_mask(self, strain_name: str) -> tuple[int, ...]:
        self.index_of(strain_name)
        return self.introduction_age_masks[strain_name]

    def introduction_age_mask_matrix(self, dtype: Any = jnp.int32) -> Any:
        if self.n_age_bins_for_introduction == 0:
            return jnp.zeros((self.n_strains, 0), dtype=dtype)

        return jnp.asarray(
            [
                self.introduction_age_masks[strain_name]
                for strain_name in self.strain_names
            ],
            dtype=dtype,
        )
