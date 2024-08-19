"""
An Abstract Class used to set up Parameters for running in Ordinary Differential Equations.

Responsible for loading and assembling functions to describe vaccination uptake, seasonality,
external transmission of new or existing viruses and other generic respiratory virus aspects.
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

import mechanistic_model.utils as utils
from mechanistic_model import SEIC_Compartments
from mechanistic_model.mechanistic_runner import MechanisticRunner


class AbstractParameters:
    @abstractmethod
    def __init__(self, parameters_config):
        # add these for mypy type checker
        self.config = {}
        self.INITIAL_STATE = tuple()
        pass

    def _solve_runner(
        self, parameters: dict, tf: int, runner: MechanisticRunner
    ) -> Solution:
        """runs the runner for `tf` days using parameters defined in `parameters`
        returning a Diffrax Solution object

        Parameters
        ----------
        parameters : dict
            parameters object containing parameters required by the runner ODEs
        tf : int
            number of days to run the runner for
        runner : MechanisticRunner
            runner class designated with solving ODEs

        Returns
        -------
        Solution
            diffrax solution object returned from runner.run()
        """
        if "INITIAL_INFECTIONS_SCALE" in parameters.keys():
            initial_state = self.scale_initial_infections(
                parameters["INITIAL_INFECTIONS_SCALE"]
            )
        else:
            initial_state = self.INITIAL_STATE
        solution = runner.run(
            initial_state,
            args=parameters,
            tf=tf,
        )
        return solution

    def get_parameters(self) -> dict:
        """
        Goes through the parameters passed to the inferer, if they are distributions, it samples them.
        Otherwise it returns their raw values.

        Converts all list types with sampled values to jax tracers.

        Returns
        -----------
        dict{str:obj} where obj may either be a float value,
        or a jax tracer, in the case of a sampled value or list containing sampled values.
        """
        # multiple chains of MCMC calling get_parameters()
        # should not share references, deep copy, GH issue for this created
        freeze_params = copy.deepcopy(self.config)
        parameters = {
            "INIT_DATE": freeze_params.INIT_DATE,
            "CONTACT_MATRIX": freeze_params.CONTACT_MATRIX,
            "POPULATION": freeze_params.POPULATION,
            "NUM_STRAINS": freeze_params.NUM_STRAINS,
            "NUM_AGE_GROUPS": freeze_params.NUM_AGE_GROUPS,
            "NUM_WANING_COMPARTMENTS": freeze_params.NUM_WANING_COMPARTMENTS,
            "WANING_PROTECTIONS": freeze_params.WANING_PROTECTIONS,
            "MAX_VACCINATION_COUNT": freeze_params.MAX_VACCINATION_COUNT,
            "STRAIN_INTERACTIONS": freeze_params.STRAIN_INTERACTIONS,
            "VACCINE_EFF_MATRIX": freeze_params.VACCINE_EFF_MATRIX,
            "BETA_TIMES": freeze_params.BETA_TIMES,
            "STRAIN_R0s": freeze_params.STRAIN_R0s,
            "INFECTIOUS_PERIOD": freeze_params.INFECTIOUS_PERIOD,
            "EXPOSED_TO_INFECTIOUS": freeze_params.EXPOSED_TO_INFECTIOUS,
            "INTRODUCTION_TIMES": freeze_params.INTRODUCTION_TIMES,
            "INTRODUCTION_SCALES": freeze_params.INTRODUCTION_SCALES,
            "INTRODUCTION_PCTS": freeze_params.INTRODUCTION_PCTS,
            "INITIAL_INFECTIONS_SCALE": freeze_params.INITIAL_INFECTIONS_SCALE,
            "CONSTANT_STEP_SIZE": freeze_params.CONSTANT_STEP_SIZE,
            "SEASONALITY_AMPLITUDE": freeze_params.SEASONALITY_AMPLITUDE,
            "SEASONALITY_SECOND_WAVE": freeze_params.SEASONALITY_SECOND_WAVE,
            "SEASONALITY_SHIFT": freeze_params.SEASONALITY_SHIFT,
            "MIN_HOMOLOGOUS_IMMUNITY": freeze_params.MIN_HOMOLOGOUS_IMMUNITY,
        }
        parameters = utils.sample_if_distribution(parameters)
        # re-create the CROSSIMMUNITY_MATRIX since we may be sampling the STRAIN_INTERACTIONS matrix now
        parameters[
            "CROSSIMMUNITY_MATRIX"
        ] = utils.strain_interaction_to_cross_immunity(
            freeze_params.NUM_STRAINS, parameters["STRAIN_INTERACTIONS"]
        )
        # create parameters based on other possibly sampled parameters
        beta = parameters["STRAIN_R0s"] / parameters["INFECTIOUS_PERIOD"]
        gamma = 1 / parameters["INFECTIOUS_PERIOD"]
        sigma = 1 / parameters["EXPOSED_TO_INFECTIOUS"]
        # last waning time is zero since last compartment does not wane
        # catch a division by zero error here.
        waning_rates = np.array(
            [
                1 / waning_time if waning_time > 0 else 0
                for waning_time in freeze_params.WANING_TIMES
            ]
        )
        # allows the ODEs to just pass time as a parameter, makes them look cleaner
        external_i_function_prefilled = jax.tree_util.Partial(
            self.external_i,
            introduction_times=parameters["INTRODUCTION_TIMES"],
            introduction_scales=parameters["INTRODUCTION_SCALES"],
            introduction_pcts=parameters["INTRODUCTION_PCTS"],
        )
        # # pre-calculate the minimum value of the seasonality curves
        seasonality_function_prefilled = jax.tree_util.Partial(
            self.seasonality,
            seasonality_amplitude=parameters["SEASONALITY_AMPLITUDE"],
            seasonality_second_wave=parameters["SEASONALITY_SECOND_WAVE"],
            seasonality_shift=parameters["SEASONALITY_SHIFT"],
        )
        # add final parameters, if your model expects added parameters, add them here
        parameters = dict(
            parameters,
            **{
                "BETA": beta,
                "SIGMA": sigma,
                "GAMMA": gamma,
                "WANING_RATES": waning_rates,
                "EXTERNAL_I": external_i_function_prefilled,
                "VACCINATION_RATES": self.vaccination_rate,
                "BETA_COEF": self.beta_coef,
                "SEASONAL_VACCINATION_RESET": self.seasonal_vaccination_reset,
                "SEASONALITY": seasonality_function_prefilled,
            }
        )

        return parameters

    @partial(jax.jit, static_argnums=(0))
    def external_i(
        self,
        t: ArrayLike,
        introduction_times: jax.Array,
        introduction_scales: jax.Array,
        introduction_pcts: jax.Array,
    ) -> jax.Array:
        """
        Given some time t, returns jnp.array of shape self.INITIAL_STATE[self.config.COMPARTMENT_IDX.I] representing external infected persons
        interacting with the population. it does so by calling some function f_s(t) for each strain s.

        MUST BE CONTINUOUS AND DIFFERENTIABLE FOR ALL TIMES t.

        The stratafication of the external population is decided by the introduced strains, which are defined by
        3 parallel lists of the time they peak (`introduction_times`),
        the number of external infected individuals introduced as a % of the tracked population (`introduction_pcts`)
        and how quickly or slowly those individuals contact the tracked population (`introduction_scales`)

        Parameters
        ----------
        `t`: float as Traced<ShapedArray(float64[])>
            current time in the model, due to the just-in-time nature of Jax this float value may be contained within a
            traced array of shape () and size 1. Thus no explicit comparison should be done on "t".

        `introduction_times`: list[int] as Traced<ShapedArray(float64[])>
            a list representing the times at which external strains should be introduced, in days, after t=0 of the model
            This list is ordered inversely to self.config.STRAIN_R0s. If 2 external strains are defined, the two
            values in `introduction_times` will refer to the last 2 STRAIN_R0s, not the first two.

        `introduction_scales`: list[float] as Traced<ShapedArray(float64[])>
            a list representing the standard deviation of the curve that external strains are introduced with, in days
            This list is ordered inversely to self.config.STRAIN_R0s. If 2 external strains are defined, the two
            values in `introduction_times` will refer to the last 2 STRAIN_R0s, not the first two.

        `introduction_pcts`: list[float] as Traced<ShapedArray(float64[])>
            a list representing the proportion of each age bin in self.POPULATION[self.config.INTRODUCTION_AGE_MASK]
            that will be exposed to the introduced strain over the entire course of the introduction.
            This list is ordered inversely to self.config.STRAIN_R0s. If 2 external strains are defined, the two
            values in `introduction_times` will refer to the last 2 STRAIN_R0s, not the first two.

        Returns
        -----------
        external_i_compartment: jax.Array
            jnp.array(shape=(self.INITIAL_STATE[self.config.COMPARTMENT_IDX.I].shape)) of external individuals to the system
            interacting with susceptibles within the system, used to impact force of infection.
        """

        # define a function that returns 0 for non-introduced strains
        def zero_function(_):
            return 0

        external_i_distributions = [
            zero_function for _ in range(self.config.NUM_STRAINS)
        ]
        introduction_percentage_by_strain = [0] * self.config.NUM_STRAINS
        for introduced_strain_idx, (
            introduced_time,
            introduction_scale,
            introduction_perc,
        ) in enumerate(
            zip(introduction_times, introduction_scales, introduction_pcts)
        ):
            # earlier introduced strains earlier will be placed closer to historical strains (0 and 1)
            dist_idx = (
                self.config.NUM_STRAINS
                - self.config.NUM_INTRODUCED_STRAINS
                + introduced_strain_idx
            )
            # use a normal PDF with std dv
            external_i_distributions[dist_idx] = partial(
                pdf, loc=introduced_time, scale=introduction_scale
            )
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
        Given some time t, returns a jnp.array of shape (self.config.NUM_AGE_GROUPS, self.config.MAX_VACCINATION_COUNT + 1)
        representing the age / vax history stratified vaccination rates for an additional vaccine. Used by transmission models
        to determine vaccination rates at a particular time step.
        In the cases that your model's definition of t=0 is later the vaccination spline's definition of t=0
        use the `VACCINATION_MODEL_DAYS_SHIFT` config parameter to shift the vaccination spline's t=0 right.

        MUST BE CONTINUOUS AND DIFFERENTIABLE FOR ALL TIMES t. If you want a piecewise implementation of vax rates must declare jump points
        in the MCMC object.

        Parameters
        ----------
        t: float as Traced<ShapedArray(float64[])>
            current time in the model, due to the just-in-time nature of Jax this float value may be contained within a
            traced array of shape () and size 1. Thus no explicit comparison should be done on "t".

        Returns
        -----------
        vaccination_rates: jnp.Array
            jnp.array(shape=(self.config.NUM_AGE_GROUPS, self.config.MAX_VACCINATION_COUNT + 1)) of vaccination rates for each age bin and vax history strata.
        """
        # shifting splines if needed for multi-epochs, 0 by default
        t_added = getattr(self.config, "VACCINATION_MODEL_DAYS_SHIFT", 0)
        # default to 1.0 (unchanged) if parameter does not exist
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
        """Returns a coefficient for the beta value for cases of external impacts
        on transmission not directly accounted for in the model.
        Currently implemented via an array search with timings BETA_TIMES and coefficients BETA_COEFICIENTS

        Parameters
        ----------
        t: float as Traced<ShapedArray(float32[])>
            current time in the model. Due to the just-in-time nature of Jax this float value may be contained within a
            traced array of shape () and size 1. Thus no explicit comparison should be done on "t".

        Returns:
            Coefficient by which BETA can be multiplied to externally increase or decrease the value to account for measures or seasonal forcing.
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
        """
        Returns the seasonlity coefficient as determined by two cosine waves
        multiplied by `seasonality_peak` and `seasonality_second_wave` and shifted by `seasonality_shift` days.

        Parameters
        -----------
        t: int/Traced<ShapedArray(int)> as jax.Tracer during runtime

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
        -----------
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

    def retrieve_population_counts(self) -> None:
        """
        A wrapper function which takes calculates the age stratified population counts across all the INITIAL_STATE compartments
        (minus the book-keeping C compartment.) and stores it in the self.config.POPULATION parameter.

        We do not recieve this data exactly from the initializer, but it is trivial to recalculate.
        """
        self.config.POPULATION = np.sum(  # sum together S+E+I compartments
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

    def load_cross_immunity_matrix(self) -> None:
        """
        Loads the Crossimmunity matrix given the strain interactions matrix.
        Strain interactions matrix is a matrix of shape (num_strains, num_strains) representing the relative immune escape risk
        of those who are being challenged by a strain in dim 0 but have recovered from a strain in dim 1.
        Neither the strain interactions matrix nor the crossimmunity matrix take into account waning.

        Updates
        ----------
        self.config.CROSSIMMUNITY_MATRIX:
            updates this matrix to shape (self.config.NUM_STRAINS, self.config.NUM_PREV_INF_HIST) containing the relative immune escape
            values for each challenging strain compared to each prior immune history in the model.
        """
        self.config.CROSSIMMUNITY_MATRIX = (
            utils.strain_interaction_to_cross_immunity(
                self.config.NUM_STRAINS, self.config.STRAIN_INTERACTIONS
            )
        )

    def load_vaccination_model(self) -> None:
        """
        loads parameters of a polynomial spline vaccination model
        stratified on age bin and current vaccination status.

        Raises FileNotFoundError if directory given does not contain the state-specific
        filename. Formatted as spline_fits_state_name.csv.

        Also raises FileNotFoundError if passed non-csv or non-file paths.
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
            assert len(knot_coefficients) == len(
                knot_locations
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
        self.config.VACCINATION_MODEL_KNOTS = jnp.array(vax_knots)
        self.config.VACCINATION_MODEL_KNOT_LOCATIONS = jnp.array(
            vax_knot_locations
        )
        self.config.VACCINATION_MODEL_BASE_EQUATIONS = jnp.array(
            vax_base_equations
        )

    def seasonal_vaccination_reset(self, t: ArrayLike) -> ArrayLike:
        """
        if model implements seasonal vaccination, returns evaluation of a continuously differentiable function
        at time `t` to outflow individuals from the top most vaccination bin (functionally the seasonal tier)
        into the second highest bin.

        Example
        ----------
        if self.config.SEASONAL_VACCINATION == True

        at `t=utils.date_to_sim_day(self.config.VACCINATION_SEASON_CHANGE)` returns 1
        else returns near 0 for t far from self.config.VACCINATION_SEASON_CHANGE.

        This value of 1 is used by model ODES to outflow individuals from the top vaccination bin
        into the one below it, indicating a new vaccination season.
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

    def load_contact_matrix(self) -> None:
        """
        a wrapper function that loads a contact matrix for the USA based on mixing paterns data found here:
        https://github.com/mobs-lab/mixing-patterns

        Updates
        ----------
        `self.config.CONTACT_MATRIX` : numpy.ndarray
            a matrix of shape (self.config.NUM_AGE_GROUPS, self.config.NUM_AGE_GROUPS) with each value representing TODO
        """
        self.config.CONTACT_MATRIX = utils.load_demographic_data(
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
        a function which modifies returns a modified version of
        self.INITIAL_STATE scaling the number of initial infections by `scale_factor`.

        Preserves the ratio of the Exposed/Infectious compartment population sizes.
        Does not modified self.INITIAL_STATE, returns a copy.

        Parameters
        ----------
        scale_factor: float
            a multiplier value >=0.0.
            `scale_factor` < 1 reduces number of initial infections,
            `scale_factor` == 1.0 leaves initial infections unchanged,
            `scale_factor` > 1 increases number of initial infections.

        Returns
        ---------
        A copy of INITIAL_INFECTIONS with each compartment being scaled according to `scale_factor`
        """
        pop_counts_by_compartment = jnp.array(
            [
                jnp.sum(compartment)
                for compartment in self.INITIAL_STATE[
                    : self.config.COMPARTMENT_IDX.C
                ]
            ]
        )
        initial_infections = (
            pop_counts_by_compartment[self.config.COMPARTMENT_IDX.E]
            + pop_counts_by_compartment[self.config.COMPARTMENT_IDX.I]
        )
        initial_susceptibles = pop_counts_by_compartment[
            self.config.COMPARTMENT_IDX.S
        ]
        # total_pop_size = initial_susceptibles + initial_infections
        new_infections_size = scale_factor * initial_infections
        # negative if scale_factor < 1.0
        gained_infections = new_infections_size - initial_infections
        scale_factor_susceptible_compartment = 1 - (
            gained_infections / initial_susceptibles
        )
        # multiplying E and I by the same scale_factor preserves their relative ratio
        scale_factors = [
            scale_factor_susceptible_compartment,
            scale_factor,
            scale_factor,
            1.0,  # for the C compartment, unchanged.
        ]
        # scale each compartment and return
        initial_state = tuple(
            [
                compartment * factor
                for compartment, factor in zip(
                    self.INITIAL_STATE, scale_factors
                )
            ]
        )
        return initial_state
