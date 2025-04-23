"""A module to set up parameters for running in Ordinary Differential Equations (ODEs).

Responsible for loading and assembling functions to describe vaccination uptake,
seasonality, external transmission of new or existing viruses and other generic
respiratory virus aspects.
"""

import copy
import os
from abc import abstractmethod
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd  # type: ignore
from diffrax import Solution
from jax.scipy.stats.norm import pdf
from jax.typing import ArrayLike

from . import utils
from .config import Config
from .logging import logger
from .mechanistic_runner import MechanisticRunner
from .typing import SEIC_Compartments


class AbstractParameters:
    """A class to define a disease-agnostic parameters object for running disease models.

    Manages parameter passing, sampling and creation, as well as definition of
    seasonality, vaccination, external introductions, and external beta shifting functions.
    """

    UPSTREAM_PARAMETERS = [
        "INIT_DATE",
        "CONTACT_MATRIX",
        "POPULATION",
        "NUM_STRAINS",
        "NUM_AGE_GROUPS",
        "NUM_WANING_COMPARTMENTS",
        "WANING_PROTECTIONS",
        "MAX_VACCINATION_COUNT",
        "STRAIN_INTERACTIONS",
        "VACCINE_EFF_MATRIX",
        "BETA_TIMES",
        "STRAIN_R0s",
        "INFECTIOUS_PERIOD",
        "EXPOSED_TO_INFECTIOUS",
        "INTRODUCTION_TIMES",
        "INTRODUCTION_SCALES",
        "INTRODUCTION_PCTS",
        "INITIAL_INFECTIONS_SCALE",
        "CONSTANT_STEP_SIZE",
        "SEASONALITY_AMPLITUDE",
        "SEASONALITY_SECOND_WAVE",
        "SEASONALITY_SHIFT",
        "MIN_HOMOLOGOUS_IMMUNITY",
        "WANING_RATES",
        "SOLVER_RELATIVE_TOLERANCE",
        "SOLVER_ABSOLUTE_TOLERANCE",
        "SOLVER_MAX_STEPS",
    ]

    @abstractmethod
    def __init__(self) -> None:
        """Initialize a parameters object for passing data to ODEs."""
        # add these for mypy type checker
        self.config = Config("{}")
        initial_state = tuple(
            [jnp.arange(0), jnp.arange(0), jnp.arange(0), jnp.arange(0)]
        )
        assert len(initial_state) == 4
        self.INITIAL_STATE: SEIC_Compartments = initial_state

    def _solve_runner(
        self, parameters: dict, tf: int, runner: MechanisticRunner
    ) -> Solution:
        """
        Run the ODE solver for a specified number of days using given parameters.

        Parameters
        ----------
        parameters : dict
            Dictionary containing parameters required by the runner's ODEs.
        tf : int
            Number of days to run the ODE solver.
        runner : MechanisticRunner
            Instance of MechanisticRunner designated for solving ODEs.

        Returns
        -------
        Solution
            Diffrax solution object returned from `runner.run()`.
        """
        logger.debug("Checking INITIAL_INFECTIONS_SCALE in parameter keys.")
        if "INITIAL_INFECTIONS_SCALE" in parameters.keys():
            initial_state = self.scale_initial_infections(
                parameters["INITIAL_INFECTIONS_SCALE"]
            )
            logger.debug("Initial state retrieved.")
        else:
            logger.debug("Setting initial state from self.")
            initial_state = self.INITIAL_STATE
            logger.debug("Initial state set.")
        logger.debug("Running the runner...")
        solution = runner.run(
            initial_state,
            args=parameters,
            tf=tf,
        )
        logger.debug("Returning runner solution.")
        return solution

    def _get_upstream_parameters(self) -> dict:
        """
        Retrieve upstream parameters from the configuration, sampling any distributions.

        Returns
        -------
        dict[str, Any]
            Dictionary mapping keys to parameters within `self.UPSTREAM_PARAMETERS`.
            Values are taken from `self.config`, with numpyro.distribution objects
            sampled and replaced with JAX ArrayLike samples.

        Raises
        ------
        RuntimeError
            If any parameter in `self.UPSTREAM_PARAMETERS` is not found in `self.config`.
        """
        # multiple chains of MCMC calling get_parameters()
        # should not share references, deep copy, GH issue for this created
        freeze_params = copy.deepcopy(self.config)
        parameters = {}
        for parameter in self.UPSTREAM_PARAMETERS:
            if hasattr(freeze_params, parameter):
                parameters[parameter] = getattr(freeze_params, parameter)
            else:
                raise RuntimeError(
                    """self.config does not contain a %s parameter, either include it in the
                     configuration file used to generate this parameters object,
                     or exclude it from self.UPSTREAM_PARAMETERS"""
                    % parameter
                )
        # sample any distributions found within this dictionary
        parameters = utils.sample_if_distribution(parameters)
        return parameters

    def generate_downstream_parameters(self, parameters: dict) -> dict:
        """
        Generate downstream dependent parameters based on upstream values.

        Parameters
        ----------
        parameters : dict
            Dictionary generated by `self._get_upstream_parameters()` containing
            static or sampled values that downstream parameters may depend on.

        Returns
        -------
        dict
            Updated version of `parameters` with additional downstream parameters added.

        Raises
        ------
        RuntimeError
            If a downstream parameter cannot find the necessary upstream values within `parameters`.
        """
        try:
            # create or re-recreate parameters based on other possibly sampled parameters
            parameters["CROSSIMMUNITY_MATRIX"] = (
                utils.strain_interaction_to_cross_immunity(
                    parameters["NUM_STRAINS"],
                    parameters["STRAIN_INTERACTIONS"],
                )
            )
            beta = parameters["STRAIN_R0s"] / parameters["INFECTIOUS_PERIOD"]
            gamma = 1 / parameters["INFECTIOUS_PERIOD"]
            sigma = 1 / parameters["EXPOSED_TO_INFECTIOUS"]
            external_i_function_prefilled = jax.tree_util.Partial(
                self.external_i,
                introduction_times=parameters["INTRODUCTION_TIMES"],
                introduction_scales=parameters["INTRODUCTION_SCALES"],
                introduction_pcts=parameters["INTRODUCTION_PCTS"],
            )
            seasonality_function_prefilled = jax.tree_util.Partial(
                self.seasonality,
                seasonality_amplitude=parameters["SEASONALITY_AMPLITUDE"],
                seasonality_second_wave=parameters["SEASONALITY_SECOND_WAVE"],
                seasonality_shift=parameters["SEASONALITY_SHIFT"],
            )
        except KeyError as e:
            err_txt = """Attempted to create a downstream parameter but was unable to find
            the required upstream values within `parameters` this is likely because it was not included
            within self.UPSTREAM_PARAMETERS and was therefore not collected
            before generating the downstream params"""
            raise RuntimeError(err_txt) from e

        # add final parameters, if your model expects added parameters, add them here
        parameters = dict(
            parameters,
            **{
                "BETA": beta,
                "SIGMA": sigma,
                "GAMMA": gamma,
                "EXTERNAL_I": external_i_function_prefilled,
                "VACCINATION_RATES": self.vaccination_rate,
                "BETA_COEF": self.beta_coef,
                "SEASONAL_VACCINATION_RESET": self.seasonal_vaccination_reset,
                "SEASONALITY": seasonality_function_prefilled,
            },
        )

        return parameters

    def get_parameters(self) -> dict:
        """Sample upstream distributions and generating downstream parameters.

        Returns
        -------
        dict[str, Any]
            Dictionary containing a combination of `self.UPSTREAM_PARAMETERS` found
            in `self.config` and downstream parameters generated from
            `self.generate_downstream_parameters()`.
        """
        parameters = self._get_upstream_parameters()
        parameters = self.generate_downstream_parameters(parameters)

        return parameters

    @partial(jax.jit, static_argnums=(0))
    def external_i(
        self,
        t: ArrayLike,
        introduction_times: jax.Array,
        introduction_scales: jax.Array,
        introduction_pcts: jax.Array,
    ) -> jax.Array:
        """Calculate the number of external infected individuals interacting with the population at time t.

        Parameters
        ----------
        t : ArrayLike
            Current time in the model.

        introduction_times : jax.Array
            List representing times at which external strains should peak
            in their rate of introduction.

        introduction_scales : jax.Array
            List representing the standard deviation of the curve for introducing
            external strains, in days.

        introduction_pcts : jax.Array
            List representing the proportion of each age bin in
            `self.POPULATION[self.config.INTRODUCTION_AGE_MASK]`
            that will be exposed to the introduced strain over the entire curve.

        Returns
        -------
        jax.Array
            An array of shape matching `self.INITIAL_STATE[self.config.COMPARTMENT_IDX.I]`
            representing external individuals interacting with tracked susceptibles.

        Notes
        -----
        Use `self.config.INTRODUCTION_AGE_MASK` to select which age bins are
        affected by external populations. External populations are not tracked
        but still interact with the contact matrix, influencing spread dynamics.
        """
        external_i_distributions = [
            lambda _: 0 for _ in range(self.config.NUM_STRAINS)
        ]  # start with zeros functions
        introduction_percentage_by_strain = [0] * self.config.NUM_STRAINS
        for introduced_strain_idx, (
            introduced_time,
            introduction_scale,
            introduction_perc,
        ) in enumerate(
            zip(introduction_times, introduction_scales, introduction_pcts)
        ):
            # INTRODUCTED_STRAINS are parallel to the END of the STRAIN_R0s
            dist_idx = (
                self.config.NUM_STRAINS
                - self.config.NUM_INTRODUCED_STRAINS
                + introduced_strain_idx
            )
            # use a normal PDF with std dv
            external_i_distributions[dist_idx] = partial(
                pdf, loc=introduced_time, scale=introduction_scale
            )  # type: ignore
            introduction_percentage_by_strain[dist_idx] = introduction_perc
        # with our external_i_distributions set up, now we can execute them on `t`
        # set up our return value
        external_i_compartment = jnp.zeros(
            self.INITIAL_STATE[self.config.COMPARTMENT_IDX.I].shape
        )
        introduction_age_mask = jnp.where(
            jnp.array(self.config.INTRODUCTION_AGE_MASK),
            1,
            0,
        )
        for strain in self.config.STRAIN_IDX:
            external_i_distribution = external_i_distributions[strain]
            introduction_perc = introduction_percentage_by_strain[strain]
            external_i_compartment = external_i_compartment.at[
                introduction_age_mask, 0, 0, strain
            ].set(
                external_i_distribution(t)
                * introduction_perc
                * self.config.POPULATION[self.config.INTRODUCTION_AGE_MASK]
            )
        return external_i_compartment

    @partial(jax.jit, static_argnums=(0))
    def vaccination_rate(self, t: ArrayLike) -> jax.Array:
        """
        Calculate the instantaneous vaccination rates stratified by age and vaccination history.

        Parameters
        ----------
        t : ArrayLike
            Current time in the model.

        Returns
        -------
        jax.Array
            An array of shape (self.config.NUM_AGE_GROUPS, self.config.MAX_VACCINATION_COUNT + 1)
            representing vaccination rates for each age bin and vaccination history strata.

        Notes
        -----
        Use `self.config.VACCINATION_MODEL_DAYS_SHIFT` to adjust t=0
        specifically for this function.
        Refer to `load_vaccination_model` for details on spline definitions
        and loading.
        The function is continuous and differentiable for all times `t`.
        """
        # shifting splines if needed for multi-epochs, 0 by default
        t_added = getattr(self.config, "VACCINATION_MODEL_DAYS_SHIFT", 0)
        vaccination_rates_log = utils.evaluate_cubic_spline(
            t + t_added,
            self.config.VACCINATION_MODEL_KNOT_LOCATIONS,
            self.config.VACCINATION_MODEL_BASE_EQUATIONS,
            self.config.VACCINATION_MODEL_KNOTS,
        )
        # one of the side effects of exp() is setting exp(0) -> 1
        # we dont want this behavior in our vaccination rates obviously
        # so we find the locations of zero and save them to remask 0 -> 0 after exp() op
        zero_mask = jnp.where(vaccination_rates_log == 0, 0, 1)
        return zero_mask * jnp.exp(
            utils.evaluate_cubic_spline(
                t + t_added,
                self.config.VACCINATION_MODEL_KNOT_LOCATIONS,
                self.config.VACCINATION_MODEL_BASE_EQUATIONS,
                self.config.VACCINATION_MODEL_KNOTS,
            )
        )

    @partial(jax.jit, static_argnums=(0))
    def beta_coef(self, t: ArrayLike) -> ArrayLike:
        """
        Calculate the coefficient to modify the transmission rate based on external factors.

        Parameters
        ----------
        t : ArrayLike
            Current time in the model.

        Returns
        -------
        ArrayLike
            Coefficient by which BETA can be multiplied to
            externally increase or decrease its value.

        Examples
        --------
        Multiple values of `t` being passed at once for this example only.

        >>> self.config.BETA_COEFICIENTS
        jnp.array([-1.0, 0.0, 1.0])

        >>> self.config.BETA_TIMES
        jnp.array([25, 50])

        >>> self.beta_coef(t=[0, 24])
        [-1. -1.]

        >>> self.beta_coef(t=[25, 26, 49])
        [0. 0. 0.]

        >>> self.beta_coef(t=[50, 51, 99])
        [1. 1. 1.]

        Notes
        -----
        The function defaults to a coefficient of 1.0 if no modifications are
        specified. It uses `BETA_TIMES` and `BETA_COEFICIENTS` from the
        configuration for adjustments.
        """
        # a smart lookup function that works with JAX just in time compilation
        # if t > self.config.BETA_TIMES_i, return self.config.BETA_COEFICIENTS_i
        if hasattr(self.config, "BETA_COEFICIENTS") and hasattr(
            self.config, "BETA_TIMES"
        ):
            # this will trigger the runner to use adaptive step size with jump_ts
            return self.config.BETA_COEFICIENTS[
                jnp.maximum(0, jnp.searchsorted(self.config.BETA_TIMES, t) - 1)
            ]
        else:  # dont modify beta
            return 1.0

    def seasonality(
        self,
        t: ArrayLike,
        seasonality_amplitude: ArrayLike,
        seasonality_second_wave: ArrayLike,
        seasonality_shift: ArrayLike,
    ) -> ArrayLike:
        """Calculate seasonlity coefficient for time `t`.

        As determined by two cosine waves multiplied by `seasonality_peak` and
        `seasonality_second_wave` and shifted by `seasonality_shift` days.

        Parameters
        ----------
        t: ArrayLike
            Current model day.
        seasonality_amplitude: float/Traced<ShapedArray(float64[])>
            maximum and minimum of the combined curves,
            taking values of `1 +/-seasonality_amplitude` respectively
        seasonality_second_wave: float/Traced<ShapedArray(float64[])>
            enforced 0 <= seasonality_second_wave <= 1.0
            adjusts how pronouced the summer wave is,
            with 1.0 being equally sized winter and summer waves, and 0 being no summer wave
        seasonality_shift: float/Traced<ShapedArray(float64[])>
            horizontal shift across time in days, cant not exceed +/-(365/2)
            if seasonality_shift=0, peak occurs at t=0.

        Returns
        -------
        <ArrayLike | Float>
            Seasonality coefficient signaling an increase (>1) or decrease (<1)
            in transmission due to the impact of seasonality.
        """
        # cosine curves are defined by a cycle of 365 days begining at jan 1st
        # start by shifting the curve some number of days such that we line up with our INIT_DATE
        seasonality_shift = (
            seasonality_shift - self.config.INIT_DATE.timetuple().tm_yday
        )
        k = 2 * jnp.pi / 365.0
        # for a closed form solution to the combination of both cosine curves
        # we must split along a boundary of second (summer) wave values
        assert not isinstance(seasonality_second_wave, complex)
        cos_val = jnp.where(
            seasonality_second_wave > 0.2,
            (seasonality_second_wave - 1)
            / (4 * seasonality_second_wave + 1e-6),
            -1,
        )
        # calculate the day on which cos1 + cos2 is at minimum, scale using that
        # such that the value on that day is 1-seasonality_amplitude
        min_day = jnp.arccos(cos_val) / k + seasonality_shift
        curve_normalizing_factor = utils.season_1peak(
            min_day,
            seasonality_second_wave,
            seasonality_shift,
        ) + utils.season_2peak(
            min_day,
            seasonality_second_wave,
            seasonality_shift,
        )
        season_curve = utils.season_1peak(
            t, seasonality_second_wave, seasonality_shift
        ) + utils.season_2peak(t, seasonality_second_wave, seasonality_shift)
        return 1 + (
            seasonality_amplitude
            * (
                2
                * (season_curve - curve_normalizing_factor)
                / (1 - curve_normalizing_factor)
                - 1
            )
        )

    def retrieve_population_counts(self) -> np.ndarray:
        """Calculate the age stratified population counts across all tracked compartments.

        Excludes the book-keeping C compartment.

        Returns
        -------
        np.ndarray
            population counts of each age bin within `self.INITIAL_STATE`
        """
        return np.sum(  # sum together S+E+I compartments
            np.array(
                [
                    np.sum(
                        compartment,
                        axis=(
                            self.config.S_AXIS_IDX.hist,
                            self.config.S_AXIS_IDX.vax,
                            self.config.S_AXIS_IDX.wane,
                        ),
                    )  # sum over all but age bin axis
                    for compartment in self.INITIAL_STATE[
                        : self.config.COMPARTMENT_IDX.C
                    ]  # avoid summing the book-keeping C compartment
                ]
            ),
            axis=(0),  # sum across compartments, keep age bins
        )

    def load_cross_immunity_matrix(self) -> jax.Array:
        """Load the Crossimmunity matrix given the strain interactions matrix.

        Returns
        -------
        jax.Array
            matrix of shape (self.config.NUM_STRAINS, self.config.NUM_PREV_INF_HIST)
            containing the relative immune escape values for each challenging
            strain compared to each prior immune history in the model.

        Notes
        -----
        Strain interactions matrix is a matrix of shape
        (self.config.NUM_STRAINS, self.config.NUM_STRAINS),
        representing the relative immune escape risk of those who are being
        challenged by a strain in dim 0 but have recovered
        previously from a strain in dim 1. Neither the strain interactions
        matrix nor the crossimmunity matrix take into account waning.
        """
        return utils.strain_interaction_to_cross_immunity(
            self.config.NUM_STRAINS, self.config.STRAIN_INTERACTIONS
        )

    def load_vaccination_model(self) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Load parameters of a polynomial spline vaccination model.

        Returns
        -------
        the following are 3 parallel lists, each with leading dimensions
        `(NUM_AGE_GROUPS, MAX_VAX_COUNT+1)` identifying the vaccination spline
        from age group I and vaccination count J to vaccination count J+1.
        (indivduals vaccinated while at `MAX_VAX_COUNT` generally stay in
        the same tier, but this is ODE specific).
        VACCINATION_MODEL_KNOTS: jax.Array
            array of knot coefficients for each knot located on
            `VACCINATION_MODEL_KNOT_LOCATIONS[i][j]`
        VACCINATION_MODEL_KNOT_LOCATIONS: jax.Array
            array of knot locations by model day, with 0 indicating the knot is
            placed on self.config.INIT_DATE.
        VACCINATION_MODEL_BASE_EQUATIONS: jax.Array
            array defining the coefficients (a,b,c,d) of each
            base equation `(a + b(t) + c(t)^2 + d(t)^3)` for the spline defined
            by `VACCINATION_MODEL_KNOT_LOCATIONS[i][j]`.

        Raises
        ------
        FileNotFoundError
            if path is not a csv file or directory. Or if directory path does not contain region
            specific file matching expected naming convention.

        Notes
        -----
        Reads spline information from `self.config.VACCINATION_MODEL_DATA`,
        if path given is a directory, attempts a region-specific lookup with
        `self.config.REGIONS[0]`, using format
        `self.config.VACCINATION_MODEL_DATA/spline_fits_{region_name}`
        """
        # if the user passes a directory instead of a file path
        # check to see if the state exists in the directory and use that
        if os.path.isdir(self.config.VACCINATION_MODEL_DATA):
            vax_spline_filename = "spline_fits_%s.csv" % (
                self.config.REGIONS[0].lower().replace(" ", "_")
            )
            state_path = os.path.join(
                self.config.VACCINATION_MODEL_DATA, vax_spline_filename
            )
            if os.path.exists(state_path):
                parameters = pd.read_csv(state_path)
            else:
                raise FileNotFoundError(
                    "Directory passed to VACCINATION_MODEL_DATA parameter, "
                    "this directory does not contain %s which is the "
                    "expected state-specific vax filename"
                    % vax_spline_filename
                )
        # given a specific file to spline fits, use those
        elif os.path.isfile(self.config.VACCINATION_MODEL_DATA):
            parameters = pd.read_csv(self.config.VACCINATION_MODEL_DATA)
        else:
            raise FileNotFoundError(
                "Path given to VACCINATION_MODEL_DATA is something other than a "
                "directory or file path, got %s. Check configuration and provide "
                "a valid directory path or filepath to vaccination splines"
                % self.config.VACCINATION_MODEL_DATA
            )
        age_bins = len(parameters["age_group"].unique())
        vax_bins = len(parameters["dose"].unique())
        # change this if you start using higher degree polynomials to fit vax model
        assert age_bins == self.config.NUM_AGE_GROUPS, (
            "the number of age bins in your model does not match the input vaccination parameters, "
            + "please provide your own vaccination parameters that match, or adjust your age bins"
        )

        assert vax_bins == self.config.MAX_VACCINATION_COUNT + 1, (
            "the number of vaccination counts in your model does not match the input vaccination parameters, "
            + "please provide your own vaccination parameters that match, or adjust your age bins"
        )
        num_knots = len(
            parameters.iloc[0][
                [col for col in parameters.columns if "location" in col]
            ]
        )
        # store splines as a series of outward flows
        # an age_group x dose combo can identify its lost population via
        # the following 3 matricies in utils.evaluate_cubic_splines()
        vax_knots = np.zeros((age_bins, vax_bins, num_knots))
        vax_knot_locations = np.zeros((age_bins, vax_bins, num_knots))
        # always 4 base terms for cubic splines
        vax_base_equations = np.zeros((age_bins, vax_bins, 4))
        # each row in csv is one flow from vax_x -> vax_y for age group z
        for _, row in parameters.iterrows():
            age_group, vaccination = row["age_group"], row["dose"]
            intersect_and_ts = row[["intersect", "t", "t2", "t3"]].values
            # coef identifies coefficients, location identifies knot locations
            knot_coefficients = row[
                [col for col in parameters.columns if "coef" in col]
            ].values
            knot_locations = row[
                [col for col in parameters.columns if "location" in col]
            ].values
            # check that same number of knots as coefficients
            assert (
                len(knot_coefficients) == len(knot_locations)
            ), "number of knot_coefficients and number of knot locations found do not match"
            age_group_idx = self.config.AGE_GROUP_IDX[age_group]
            # splines `dose` dictate the `to_dose`, but we store them as outward flows
            # thus subtract 1, we dont support skipping doses
            vax_idx = vaccination - 1
            vax_base_equations[age_group_idx, vax_idx, :] = np.array(
                intersect_and_ts
            )
            vax_knots[age_group_idx, vax_idx, :] = np.array(knot_coefficients)
            vax_knot_locations[age_group_idx, vax_idx, :] = np.array(
                knot_locations
            )
        return (
            jnp.array(vax_knots),
            jnp.array(vax_knot_locations),
            jnp.array(vax_base_equations),
        )

    def seasonal_vaccination_reset(self, t: ArrayLike) -> ArrayLike:
        """Calculate seasonal vaccination outflow coefficient.

        If model implements seasonal vaccination, returns evaluation of a
        continuously differentiable function at time `t` to outflow individuals
        from the top most vaccination bin (functionally the seasonal tier)
        into the second highest bin.

        Parameters
        ----------
        t: ArrayLike
            current time in the model.

        Examples
        --------
        >>> assert self.config.SEASONAL_VACCINATION
        >>> self.config.VACCINATION_SEASON_CHANGE
        50
        >>> np.isclose(self.seasonal_vaccination_reset(50), 1.0)
        True
        >>> np.isclose(self.seasonal_vaccination_reset(49, 51), [1.0, 1.0])
        [False, False]
        """
        if (
            hasattr(self.config, "SEASONAL_VACCINATION")
            and self.config.SEASONAL_VACCINATION
        ):
            # outflow function must be positive if and only if
            # it is time to move people from seasonal bin back to max ordinal bin
            # use a sine wave that occurs once a year to achieve this effect
            peak_of_function = 182.5
            # shift this value using shift_t to align with self.config.VACCINATION_SEASON_CHANGE
            # such that outflow_fn(self.config.VACCINATION_SEASON_CHANGE) == 1.0 always
            shift_t = (
                peak_of_function
                - (
                    self.config.VACCINATION_SEASON_CHANGE
                    - self.config.INIT_DATE
                ).days
            )
            # raise to an even exponent to remove negatives,
            # pick 1000 since too high of a value likely to be stepped over by adaptive step size
            # divide by 730 so wave only occurs 1 per every 365 days
            # multiply by 2pi since we pass days as int
            return jnp.sin((2 * jnp.pi * (t + shift_t) / 730)) ** 1000
        else:
            # if no seasonal vaccination, this function always returns zero
            return 0

    def load_contact_matrix(self) -> np.ndarray:
        """Load region specific contact matrix.

        Usually sourced from https://github.com/mobs-lab/mixing-patterns.

        Returns
        -------
        numpy.ndarray
            a matrix of shape (self.config.NUM_AGE_GROUPS, self.config.NUM_AGE_GROUPS)
            where `CONTACT_MATRIX[i][j]` refers to the per capita
            interaction rate between age bin `i` and `j`
        """
        return utils.load_demographic_data(
            self.config.DEMOGRAPHIC_DATA_PATH,
            self.config.REGIONS,
            self.config.NUM_AGE_GROUPS,
            self.config.AGE_LIMITS[0],
            self.config.AGE_LIMITS,
        )[self.config.REGIONS[0]]["avg_CM"]

    def scale_initial_infections(
        self, scale_factor: ArrayLike
    ) -> SEIC_Compartments:
        """
        Scale the number of initial infections by a specified factor.

        This function returns a modified version of `self.INITIAL_STATE`,
        scaling the number of initial infections while preserving the ratio
        between the Exposed and Infectious compartments. The original
        `self.INITIAL_STATE` remains unchanged.

        Parameters
        ----------
        scale_factor : float
            A multiplier value >= 0.0.
            - `scale_factor < 1`: Reduces the number of initial infections.
            - `scale_factor == 1.0`: Leaves initial infections unchanged.
            - `scale_factor > 1`: Increases the number of initial infections.

        Returns
        -------
        SEIC_Compartments
            A copy of `self.INITIAL_STATE` with each compartment
            scaled up or down depending on `scale_factor`.

        Notes
        -----
        The function ensures that the relative sizes of
        Exposed and Infectious compartments are preserved during scaling.
        """
        # negative if scale_factor < 1.0
        e_new = self.INITIAL_STATE[1] * scale_factor
        i_new = self.INITIAL_STATE[2] * scale_factor
        age_stratified_infections_delta = jnp.sum(
            e_new + i_new,
            axis=(1, 2, 3),
        ) - jnp.sum(
            self.INITIAL_STATE[1] + self.INITIAL_STATE[2],
            axis=(1, 2, 3),
        )
        age_stratified_infections_scaling = 1 - (
            age_stratified_infections_delta
            / jnp.sum(
                self.INITIAL_STATE[0],
                axis=(1, 2, 3),
            )
        )
        # multiplying E and I by the same scale_factor preserves their relative ratio
        s_new = (
            age_stratified_infections_scaling[:, None, None, None]
            * self.INITIAL_STATE[0]
        )
        return (
            s_new,
            e_new,
            i_new,
            self.INITIAL_STATE[-1],
        )
