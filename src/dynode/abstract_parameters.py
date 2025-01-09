"""
A module containing an abstract class used to set up parameters for
running in Ordinary Differential Equations (ODEs).

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

from . import SEIC_Compartments, utils
from .config import Config
from .mechanistic_runner import MechanisticRunner


class AbstractParameters:
    """A class to define a disease-agnostic parameters object for running
    disease models. Manages parameter passing, sampling and creation,
    as well as definition of seasonality, vaccination,
    external introductions, and external beta shifting functions
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
        """runs the runner (ode-solver) for `tf` days using
        parameters defined in `parameters` returning a Diffrax Solution object

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
            diffrax solution object returned from `runner.run()`
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

    def _get_upstream_parameters(self) -> dict:
        """
        returns a dictionary containing self.UPSTREAM_PARAMETERS,
        erroring if any of the parameters are not in self.config.

        Samples any parameters which are of type(numpyro.distribution).

        Returns
        ------------
        dict[str: Any]

        returns a dictionary where keys map to parameters within
        self.UPSTREAM_PARAMETERS and the values are the value of that
        parameter within self.config. numpyro.distribution objects are sampled
        and replaced with a jax ArrayLike sample from that distribution.
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
        """takes an existing parameters object and attempts to generate
        downstream dependent parameters, based on the values contained within
        `parameters`.

        Raises RuntimeError if a downstream parameter
        does not find the necessary values it needs within `parameters`

        Parameters
        ----------
        parameters : dict
            parameters dictionary generated by `self._get_upstream_parameters()`
            containing static or sampled values on which
            downstream parameters may depend

        Returns
        -------
        dict
            an appended onto version of `parameters` with additional downstream parameters added.
        """
        try:
            # create or re-recreate parameters based on other possibly sampled parameters
            parameters[
                "CROSSIMMUNITY_MATRIX"
            ] = utils.strain_interaction_to_cross_immunity(
                parameters["NUM_STRAINS"],
                parameters["STRAIN_INTERACTIONS"],
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
            }
        )

        return parameters

    def get_parameters(self) -> dict:
        """
        Goes through parameters listed in self.UPSTREAM_PARAMETERS,
        sampling them if they are distributions.
        Then generates any downstream parameters that rely on those parameters
        in self.generate_downstream_parameters().
        Returning the resulting dictionary for use in ODEs

        Returns
        -----------
        dict{str, Any}
            dict containing a combination of `self.UPSTREAM_PARAMETERS` found
            in `self.config` and downstream parameters from
            `self.generate_downstream_parameters()`
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
        """
        Given some time t, returns jnp.array of shape
        `self.INITIAL_STATE[self.config.COMPARTMENT_IDX.I]` representing
        external infected persons interacting with the population.

        CONTINUOUS AND DIFFERENTIABLE FOR ALL TIMES t.

        The stratafication of the external population is decided by the
        introduced strains, which are defined by 3 parallel lists.
        peak intro rate date (`introduction_times`),
        how quickly or slowly those individuals contact the tracked population
        (`introduction_scales`), and the magnitude of external
        infected individuals introduced as a % of the tracked population
        (`introduction_pcts`)

        Parameters
        ----------
        `t`: ArrayLike
            current time in the model.

        `introduction_times`: jax.Array
            a list representing the times at which external strains should peak
            in their rate of external introduction.
            if `len(introduction_times) < len(self.config.STRAIN_R0s)` earlier
            strains are not introduced.

        `introduction_scales`:jax.Array
            a list representing the standard deviation of the
            curve that external strains are introduced with, in days

        `introduction_pcts`: jax.Array
            a list representing the proportion of each age bin in
            self.POPULATION[self.config.INTRODUCTION_AGE_MASK] that will be
            exposed to the introduced strain over the whole curve

        Returns
        -----------
        jax.Array
            jnp.array(shape=(self.INITIAL_STATE[self.config.COMPARTMENT_IDX.I].shape))
            of external individuals to the system interacting with tracked
            susceptibles within the system, used to impact force of infection.

        Note
        -----------
        use the boolean list `self.config.INTRODUCTION_AGE_MASK`
        to select which age bins the external populations will have.
        External populations are not tracked, but still interact with
        the contact matrix, meaning the age of external "travelers"
        is still a factor in the spread of a new strain.
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
            # INTRODUCTED_STRAINS are parallel to the END of the STRAIN_R0s
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
        Given some time t, returns a jnp.array of shape
        (self.config.NUM_AGE_GROUPS, self.config.MAX_VACCINATION_COUNT + 1)
        representing the instantaneous age / vax history stratified vaccination
        rates for an additional vaccine.

        CONTINUOUS AND DIFFERENTIABLE FOR ALL TIMES `t`.

        Parameters
        ----------
        t: ArrayLike
            current time in the model.

        Returns
        -----------
        vaccination_rates: jnp.Array
            `shape=(self.config.NUM_AGE_GROUPS, self.config.MAX_VACCINATION_COUNT + 1) `
            of vaccination rates for each age bin and vax history strata.

        Note
        -----------
        use `self.config.VACCINATION_MODEL_DAYS_SHIFT` param to shift t=0 for
        specifically the vaccination_rates() function and not the whole model.

        see `load_vaccination_model` for description on spline definitions and
        loading.
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
        """Mechanism to directly influence transmission rate to account for
        external factors. Defaults to 1.0. Modified via the `BETA_TIMES` and
        `BETA_COEFICIENTS` config parameters. See Example for
        behavior.

        Parameters
        ----------
        t: ArrayLike
            current time in the model.

        Returns
        ----------
            ArrayLike
            Coefficient by which BETA can be multiplied to externally
            increase or decrease the value.

        Example
        ----------
        multiple values of `t` passed in at once to save space.
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
        Strain interactions matrix is a matrix of shape
        (self.config.NUM_STRAINS, self.config.NUM_STRAINS)
        representing the relative immune escape risk of those who are being
        challenged by a strain in dim 0 but have recovered
        previously from a strain in dim 1. Neither the strain interactions matrix
        nor the crossimmunity matrix take into account waning.

        Updates
        ----------
        self.config.CROSSIMMUNITY_MATRIX:
            updates this matrix to shape
            (self.config.NUM_STRAINS, self.config.NUM_PREV_INF_HIST)
            containing the relative immune escape values for each challenging
            strain compared to each prior immune history in the model.
        """
        self.config.CROSSIMMUNITY_MATRIX = (
            utils.strain_interaction_to_cross_immunity(
                self.config.NUM_STRAINS, self.config.STRAIN_INTERACTIONS
            )
        )

    def load_vaccination_model(self) -> None:
        """
        loads parameters of a polynomial spline vaccination model
        stratified on age bin and current vaccination status. Reads spline
        information from `self.config.VACCINATION_MODEL_DATA`, if path given
        is a directory, attempts a region-specific lookup with
        `self.config.REGIONS[0]`, using format
        `self.config.VACCINATION_MODEL_DATA/spline_fits_{region_name}`

        Raises `FileNotFoundError` if path is not a csv file or directory.
        Raises `FileNotFoundError` if directory path does not contain region
        specific file matching expected naming convention.

        UPDATES
        -----------
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
        if model implements seasonal vaccination, returns evaluation of a
        continuously differentiable function at time `t` to outflow individuals
        from the top most vaccination bin (functionally the seasonal tier)
        into the second highest bin.

        Parameters
        ----------
        t: ArrayLike
            current time in the model.

        Example
        ----------
        if self.config.SEASONAL_VACCINATION == True

        at `t=utils.date_to_sim_day(self.config.VACCINATION_SEASON_CHANGE)` returns 1
        else returns near 0 for t far from self.config.VACCINATION_SEASON_CHANGE.

        This value of 1 is used by model ODES to outflow individuals
        from the top vaccination bin into the one below it,
        indicating a new vaccination season.
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
        loads region specific contact matrix, usually sourced from
        https://github.com/mobs-lab/mixing-patterns

        Updates
        ----------
        `self.config.CONTACT_MATRIX` : numpy.ndarray
            a matrix of shape (self.config.NUM_AGE_GROUPS, self.config.NUM_AGE_GROUPS)
            where `CONTACT_MATRIX[i][j]` refers to the per capita
            interaction rate between age bin `i` and `j`
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
            `scale_factor < 1` reduces number of initial infections,
            `scale_factor == 1.0` leaves initial infections unchanged,
            `scale_factor > 1` increases number of initial infections.

        Returns
        ---------
        A copy of `self.INITIAL_INFECTIONS` with each compartment
        scaled according to `scale_factor`
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
        assert len(initial_state) == 4
        return initial_state
