from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class CompileOptions:
    """
    Options controlling static runtime compilation.

    These are intentionally conservative. The compiler should catch as many
    static errors as possible before NumPyro/JAX/Diffrax execution begins.
    """

    run_spec_validation_hooks: bool = True
    validate_runtime_layout: bool = True
    validate_parameter_dependencies: bool = True
    validate_data_dependencies: bool = True
    validate_transmission_dependencies: bool = True
    validate_initializer_dependencies: bool = True
    validate_data_spec: bool = True

    # For the current architecture, prior distribution parameters may depend on
    # earlier sampled priors, but not on deterministic parameters. Deterministic
    # parameters are resolved after priors.
    allow_prior_dependencies_on_deterministics: bool = False

    # Optional metadata merged into RuntimeModel.metadata.
    metadata: Mapping[str, str] = field(default_factory=dict)

    # Names supplied by an outer experiment context. These are accepted when
    # validating model-local references.
    external_parameter_names: frozenset[str] = field(default_factory=frozenset)
