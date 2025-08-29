"""Module containing Parameter classes for storing DynODE parameters."""

from typing import List, Union

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

from dynode.infer import sample_then_resolve

from .deterministic_parameter import DeterministicParameter
from .simulation_config import SimulationConfig
from .strains import Strain


class Parameter:
    name: str
    distribution: Union[Distribution, float] = None
    deterministic: DeterministicParameter = None


class ParameterSet(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    parameters: list[Parameter]


class CompartmentalModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    shared_parameters: ParameterSet  # add Pydantic Field to class attributes
    configs: dict[int, SimulationConfig]

    def model_post_init(self, __context) -> None:
        self.shared_parameters = sample_then_resolve(self.shared_parameters)

        for _, config in self.configs.items():
            config.inject_parameters(parameter_set=self.shared_parameters)
            config.sample_then_resolve_parameters()

    def numpyro_model(self, **kwargs):
        """User must implement this method to define the NumPyro model."""

        raise NotImplementedError(
            "implement functionality to get initial state"
        )


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
    strain_interactions: dict[
        str,
        dict[
            str,
            NonNegativeFloat
            | ArrayLike
            | Distribution
            | DeterministicParameter,
        ],
    ]
    strains: List[Strain]

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


class Params(BaseModel):
    """Miscellaneous parameters of an ODE model."""

    # allow users to pass custom types to ParamStore
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    solver_params: SolverParams
    transmission_params: TransmissionParams
