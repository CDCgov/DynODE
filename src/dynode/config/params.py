"""Module containing Parameter classes for storing DynODE parameters."""

from typing import Dict, List, Union

import numpy as np
from diffrax import AbstractSolver, Tsit5
from jax.typing import ArrayLike
from numpyro.distributions import Distribution
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
    field_validator,
    model_validator,
)
from typing_extensions import Self

from .deterministic_parameter import DeterministicParameter
from .parameter_set import ParameterSet
from .strains import Strain


class SolverParams(ParameterSet):
    """Parameters used by the ODE solver."""

    # model_config = ConfigDict(arbitrary_types_allowed=True)
    solver_method: AbstractSolver = Field(
        default_factory=lambda: Tsit5(),
        description="""What sort of differential equation solver you wish to
        use to solve ODEs, defaults to Tsit5(), a general solver good for
        non-stiff problems. For more information on picking a solver see:
        https://docs.kidger.site/diffrax/usage/how-to-choose-a-solver/""",
    )
    ode_solver_rel_tolerance: PositiveFloat = Field(
        default=1e-5,
        description="""Solver relative tolerance, used for adaptive step sizer
        to decide the size of a subsequent step. Use constant_step_size to
        switch to constant solver mode. For more information on tolerance see
        the `choosing tolerances` drop down here:
        https://docs.kidger.site/diffrax/api/stepsize_controller/#diffrax.PIDController""",
    )
    ode_solver_abs_tolerance: PositiveFloat = Field(
        default=1e-6,
        description="""Solver absolute tolerance, used for adaptive step sizer
        to decide the size of a subsequent step. Use constant_step_size to
        switch to constant solver mode. For more information on tolerance see
        the `choosing tolerances` drop down here:
        https://docs.kidger.site/diffrax/api/stepsize_controller/#diffrax.PIDController""",
    )
    max_steps: PositiveInt = Field(
        default=int(1e6),
        description="""The maximum number of steps the ode solver will take
        before raising an error. For complex problems use higher number.""",
    )
    constant_step_size: NonNegativeFloat = Field(
        default=0,
        description="""If non-zero, solver will use constant step size
        equal to the value set. If 0 solver will use adaptive step size with
        ode_solver_rel/abs_tolerance""",
    )
    discontinuity_points: list[float] = Field(
        default_factory=lambda: [],
        description="""Points in the ode's solve that a discontinuity occurs,
        meaning the higher order gradiants are not smooth. Float values
        represent the simulation day, or days since init date of the model.""",
    )


class TransmissionParams(ParameterSet):
    """Transmission Parameters for the respiratory model."""

    # model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    #    strain_interactions: dict[
    #        str,
    #        dict[
    #            str,
    #            NonNegativeFloat
    #            | ArrayLike
    #            | Distribution
    #            | DeterministicParameter,
    #        ],
    #    ]

    # Default off-diagonal fallback (used when a pair is not specified on either strain)
    # This preserves your old behavior if you set it to DeterministicParameter('crossimmunity')
    default_offdiag: Union[
        float, ArrayLike, Distribution, DeterministicParameter
    ] = Field(
        default=DeterministicParameter("crossimmunity"),
        description="Fallback interaction value for unspecified off-diagonal pairs.",
    )

    # Optional: enforce diagonal to be exactly 1.0 unless explicitly overridden
    force_diag_ones: bool = True

    strains: List[Strain]

    # Build a name->index map once
    @property
    def strains_to_idx(self) -> Dict[str, int]:
        return {s.strain_name: i for i, s in enumerate(self.strains)}

    @field_validator("strains", mode="before")
    @classmethod
    def _validate_strains_field_not_empty(
        cls, strains: List[Strain]
    ) -> List[Strain]:
        """Ensure that the strains field is not empty."""
        if not strains:
            raise ValueError("strains field must contain at least one Strain.")
        return strains

    @model_validator(mode="after")
    def _validate_unique_names(self) -> Self:
        names = [s.strain_name for s in self.strains]
        if len(names) != len(set(names)):
            dupes = [n for n in names if names.count(n) > 1]
            raise ValueError(f"Duplicate strain_name(s): {sorted(set(dupes))}")
        return self

    @model_validator(mode="after")
    def _validate_strain_interactions(self) -> Self:
        strain_names = [strain.strain_name for strain in self.strains]
        # check that strain_interactions contains all strains and nothing but those strains
        assert set(strain_names) == set(self.strain_interactions.keys()), (
            f"first dimension of strain_interactions must contain all strain names as "
            f"keys. Found {list(self.strain_interactions.keys())}"
            f"but expected {strain_names}."
        )

        for strain_name, interactions in self.strain_interactions.items():
            assert set(strain_names) == set(interactions.keys()), (
                f"strain_interactions[{strain_name}] interactions must contain "
                f"all strains as keys, including itself, "
                f"found {list(interactions.keys())}, expected {strain_names}."
            )
        return self

    @field_validator("strains", mode="after")
    @classmethod
    def _validate_strains_field_matching(
        cls, strains: list[Strain]
    ) -> list[Strain]:
        strain_intro_ages = [
            strain.introduction_ages
            for strain in strains
            if strain.is_introduced
        ]
        assert all(
            [compare == strain_intro_ages[0] for compare in strain_intro_ages]
        ), (
            "currently DynODE requires all strains have matching introduction_ages."
        )
        # Fields to check for consistency (excluding introduction_* fields)
        optional_fields_to_check = [
            "exposed_to_infectious",
            "vaccine_efficacy",
        ]

        # whether or not the fields exist in one or more of the Strains
        field_presence_tracker = {
            field_name: False for field_name in optional_fields_to_check
        }

        # Check each strain and update tracker
        for strain in strains:
            for field_name in optional_fields_to_check:
                if getattr(strain, field_name) is not None:
                    field_presence_tracker[field_name] = True

        # Validate that all required fields are consistently set across all strains
        for field_name in optional_fields_to_check:
            if field_presence_tracker[field_name]:
                for strain in strains:
                    if getattr(strain, field_name) is None:
                        raise AssertionError(
                            f"if {field_name} is set within one strain it must be set in all of them."
                        )

        return strains

    def build_interaction_matrix(
        self,
        as_array: bool = True,
    ) -> np.ndarray | List[List[object]]:
        """
        Returns an n×n matrix M where M[i,j] is the effect of source strain i on target strain j.
        Resolution rules (in order of precedence):
          1) If source_strain.interactions has an entry for target, use it.
          2) Else if target_strain.interactions has an entry for source and you want symmetry, you could use it (optional).
          3) Else use default_offdiag for off-diagonals.
          4) Diagonal is 1.0 if force_diag_ones, else default_offdiag (or explicit).
        """
        n = len(self.strains)
        # names = [s.strain_name for s in self.strains]
        M: List[List[object]] = [[None] * n for _ in range(n)]

        for i, s_i in enumerate(self.strains):
            for j, s_j in enumerate(self.strains):
                if i == j:
                    val = s_i.interactions.get(
                        s_i.strain_name,
                        1.0 if self.force_diag_ones else self.default_offdiag,
                    )
                else:
                    # priority: source's explicit -> (optional) target's explicit -> fallback
                    if s_j.strain_name in s_i.interactions:
                        val = s_i.interactions[s_j.strain_name]
                    else:
                        # if you want to allow implicit symmetry, uncomment next two lines:
                        # if s_i.strain_name in s_j.interactions:
                        #     val = s_j.interactions[s_i.strain_name]
                        val = self.default_offdiag
                M[i][j] = val

        if as_array:
            # Return as an object array to hold floats / jnp / Distributions / DeterministicParameter
            return np.array(M, dtype=object)
        return


class ParameterWrapper(BaseModel):
    """Miscellaneous parameters of an ODE model."""

    # allow users to pass custom types to ParamStore
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    solver_params: SolverParams
    transmission_params: TransmissionParams
    distributions: ParameterSet
    deterministic_params: ParameterSet
    # strains: ParameterSet
    # compartments: ParameterSet
