from abc import abstractmethod
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax.scipy.stats.norm import pdf

import utils


class AbstractParameters:
    @abstractmethod
    def __init__(self, parameters_config):
        pass

    @abstractmethod
    def get_parameters(self) -> dict:
        """
        Returns parameters as floats, lists, or jax dynamic tracers to be used in model ODEs
        """
        pass

    @partial(jax.jit, static_argnums=(0))
    def external_i(
        self, t, introduction_times, introduction_scales, introduction_percs
    ):
        """
        Given some time t, returns jnp.array of shape self.INITIAL_STATE[self.config.COMPARTMENT_IDX.I] representing external infected persons
        interacting with the population. it does so by calling some function f_s(t) for each strain s.

        MUST BE CONTINUOUS AND DIFFERENTIABLE FOR ALL TIMES t.

        The stratafication of the external population is decided by the introduced strains, which are defined by
        3 parallel lists of the time they peak (`introduction_times`),
        the number of external infected individuals introduced as a % of the tracked population (`introduction_percs`)
        and how quickly or slowly those individuals contact the tracked population (`introduction_scales`)

        Parameters
        ----------
        `t`: float as Traced<ShapedArray(float32[])>
            current time in the model, due to the just-in-time nature of Jax this float value may be contained within a
            traced array of shape () and size 1. Thus no explicit comparison should be done on "t".

        `introduction_times`: list[int]
            a list representing the times at which external strains should be introduced, in days, after t=0 of the model
            This list is ordered inversely to self.config.STRAIN_R0s. If 2 external strains are defined, the two
            values in `introduction_times` will refer to the last 2 STRAIN_R0s, not the first two.

        `introduction_scales`: list[float]
            a list representing the standard deviation of the curve that external strains are introduced with, in days
            This list is ordered inversely to self.config.STRAIN_R0s. If 2 external strains are defined, the two
            values in `introduction_times` will refer to the last 2 STRAIN_R0s, not the first two.

        `introduction_percs`: list[float]
            a list representing the proportion of each age bin in self.POPULATION[self.config.INTRODUCTION_AGE_MASK]
            that will be exposed to the introduced strain over the entire course of the introduction.
            This list is ordered inversely to self.config.STRAIN_R0s. If 2 external strains are defined, the two
            values in `introduction_times` will refer to the last 2 STRAIN_R0s, not the first two.

        Returns
        -----------
        external_i_compartment: jnp.array()
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
            zip(introduction_times, introduction_scales, introduction_percs)
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
    def vaccination_rate(self, t):
        """
        Given some time t, returns a jnp.array of shape (self.config.NUM_AGE_GROUPS, self.config.MAX_VAX_COUNT + 1)
        representing the age / vax history stratified vaccination rates for an additional vaccine. Used by transmission models
        to determine vaccination rates at a particular time step.
        In the cases that your model's definition of t=0 is later the vaccination spline's definition of t=0
        use the `VAX_MODEL_DAYS_SHIFT` config parameter to shift the vaccination spline's t=0 right.

        MUST BE CONTINUOUS AND DIFFERENTIABLE FOR ALL TIMES t. If you want a piecewise implementation of vax rates must declare jump points
        in the MCMC object.

        Parameters
        ----------
        t: float as Traced<ShapedArray(float32[])>
            current time in the model, due to the just-in-time nature of Jax this float value may be contained within a
            traced array of shape () and size 1. Thus no explicit comparison should be done on "t".

        Returns
        -----------
        vaccination_rates: jnp.array()
            jnp.array(shape=(self.config.NUM_AGE_GROUPS, self.config.MAX_VAX_COUNT + 1)) of vaccination rates for each age bin and vax history strata.
        """
        t_added = getattr(self.config, "VAX_MODEL_DAYS_SHIFT", 0)
        return jnp.exp(
            utils.evaluate_cubic_spline(
                t + t_added,
                self.config.VAX_MODEL_KNOT_LOCATIONS,
                self.config.VAX_MODEL_BASE_EQUATIONS,
                self.config.VAX_MODEL_KNOTS,
            )
        )

    @partial(jax.jit, static_argnums=(0))
    def beta_coef(self, t):
        """Returns a coefficient for the beta value for cases of seasonal forcing or external impacts
        onto beta not directly measured in the model. e.g., masking mandates or holidays.
        Currently implemented via an array search with timings BETA_TIMES and coefficients BETA_COEFICIENTS

        Parameters
        ----------
        t: float as Traced<ShapedArray(float32[])>
            current time in the model. Due to the just-in-time nature of Jax this float value may be contained within a
            traced array of shape () and size 1. Thus no explicit comparison should be done on "t".

        Returns:
        coefficient by which BETA can be multiplied to externally increase or decrease the value to account for measures or seasonal forcing.
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
        t,
        seasonality_amplitude,
        seasonality_second_wave,
        seasonality_shift,
    ):
        """Returns the seasonlity coefficient for Beta as determined by two cosine waves
        multiplied by `seasonality_peak` and `seasonality_second_wave` and shifted by `seasonality_shift` days.

        Parameters
        -----------
        t: int

        seasonality_amplitude: float/Traced<ShapedArray(float32[])>
            maximum and minimum of the combined curves,
            taking values of `+/-seasonality_amplitude` respectively
        seasonality_second_wave: float/Traced<ShapedArray(float32[])>
            enforced 0 <= seasonality_second_wave <= 1.0
            adjusts how pronouced the summer wave is,
            with 1.0 being equally sized winter and summer waves, and 0 being no summer wave
        seasonality_shift: float/Traced<ShapedArray(float32[])>
            horizontal shift across time in days, cant not exceed +/-(365/2)
            if seasonality_shift=0, peak occurs at t=0.
        m: float/Traced<ShapedArray(float32[])>
            the minimum value of the two cosine curves over the year
            for optimal efficiency this is computed in advance

        """
        # shift our curve an additional number of days to account for t=0=INIT_DATE and not jan 1st
        seasonality_shift = (
            seasonality_shift - self.config.INIT_DATE.timetuple().tm_yday
        )
        m = float("inf")
        for t1 in range(365):
            m = jnp.minimum(
                m,
                utils.season_1peak(
                    t1,
                    seasonality_second_wave,
                    seasonality_shift,
                )
                + utils.season_2peak(
                    t1,
                    seasonality_second_wave,
                    seasonality_shift,
                ),
            )

        season_curve = utils.season_1peak(
            t, seasonality_second_wave, seasonality_shift
        ) + utils.season_2peak(t, seasonality_second_wave, seasonality_shift)
        return 1 + (
            seasonality_amplitude * (2 * (season_curve - m) / (1 - m) - 1)
        )

    def retrieve_population_counts(self):
        """
        A wrapper function which takes retrieves the age stratified population counts across all the INITIAL_STATE compartments
        (minus the book-keeping C compartment.)
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

    def load_cross_immunity_matrix(self):
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

    def load_vaccination_model(self):
        """
        loads parameters of a polynomial spline vaccination model stratified on age bin and current vaccination status.
        also loads in the spline knot locations.
        """
        parameters = pd.read_csv(self.config.VAX_MODEL_DATA)
        age_bins = len(parameters["age_group"].unique())
        vax_bins = len(parameters["dose"].unique())
        # change this if you start using higher degree polynomials to fit vax model
        assert age_bins == self.config.NUM_AGE_GROUPS, (
            "the number of age bins in your model does not match the input vaccination parameters, "
            + "please provide your own vaccination parameters that match, or adjust your age bins"
        )

        assert vax_bins == self.config.MAX_VAX_COUNT + 1, (
            "the number of vaccination counts in your model does not match the input vaccination parameters, "
            + "please provide your own vaccination parameters that match, or adjust your age bins"
        )
        vax_knots = np.zeros(
            (age_bins, vax_bins, self.config.VAX_MODEL_NUM_KNOTS)
        )
        vax_knot_locations = np.zeros(
            (age_bins, vax_bins, self.config.VAX_MODEL_NUM_KNOTS)
        )
        vax_base_equations = np.zeros((age_bins, vax_bins, 4))  # always 4
        for _, row in parameters.iterrows():
            age_group, vaccination = row["age_group"], row["dose"]
            intersect_and_ts = row[["intersect", "t", "t2", "t3"]].values
            knot_coefficients = row[
                [col for col in parameters.columns if "coef" in col]
            ].values
            knot_locations = row[
                [col for col in parameters.columns if "location" in col]
            ].values
            assert len(knot_coefficients) == len(
                knot_locations
            ), "number of knot_coefficients and number of knot locations found do not match"
            assert len(knot_coefficients) == self.config.VAX_MODEL_NUM_KNOTS, (
                "number of knots found in %s does not match number specified in self.config.VAX_MODEL_NUM_KNOTS"
                % self.config.VAX_MODEL_DATA
            )
            age_group_idx = self.config.AGE_GROUP_IDX[age_group]
            vax_idx = vaccination - 1
            vax_base_equations[age_group_idx, vax_idx, :] = np.array(
                intersect_and_ts
            )
            vax_knots[age_group_idx, vax_idx, :] = np.array(knot_coefficients)
            vax_knot_locations[age_group_idx, vax_idx, :] = np.array(
                knot_locations
            )
        self.config.VAX_MODEL_KNOTS = jnp.array(vax_knots)
        self.config.VAX_MODEL_KNOT_LOCATIONS = jnp.array(vax_knot_locations)
        self.config.VAX_MODEL_BASE_EQUATIONS = jnp.array(vax_base_equations)

    def seasonal_vaccination_reset(self, t):
        """
        if model implements seasonal vaccination, returns evaluation of a continuously differentiable function
        at x=t to outflow individuals from the top most vaccination bin (labeled the seasonal bin)
        into the second highest bin.

        Example
        ----------
        if self.config.SEASONAL_VACCINATION == True

        at `t=utils.date_to_sim_day(self.config.VAX_SEASON_CHANGE)` returns 1
        else returns near 0 for t far from self.config.VAX_SEASON_CHANGE.
        """
        if (
            hasattr(self.config, "SEASONAL_VACCINATION")
            and self.config.SEASONAL_VACCINATION
        ):
            # outflow function must be positive if and only if
            # it is time to move people from seasonal bin back to max ordinal bin
            # use a sine wave that occurs once a year to achieve this effect
            peak_of_function = 182.5
            # shift this value using shift_t to align with self.config.VAX_SEASON_CHANGE
            # such that outflow_fn(self.config.VAX_SEASON_CHANGE) == 1.0 always
            shift_t = (
                peak_of_function
                - (self.config.VAX_SEASON_CHANGE - self.config.INIT_DATE).days
            )
            # raise to an even exponent to remove negatives,
            # pick 1000 since too high of a value likely to be stepped over by adaptive step size
            # divide by 730 so wave only occurs 1 per every 365 days
            # multiply by 2pi since we pass days as int
            return jnp.sin((2 * jnp.pi * (t + shift_t) / 730)) ** 1000
        else:
            # if no seasonal vaccination, this function always returns zero
            return 0

    def load_contact_matrix(self):
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

    def scale_initial_infections(self, scale_factor):
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
