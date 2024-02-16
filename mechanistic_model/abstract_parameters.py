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
    def external_i(self, t):
        """
        Given some time t, returns jnp.array of shape self.INITIAL_STATE[self.COMPARTMENT_IDX.I] representing external infected persons
        interacting with the population. it does so by calling some function f_s(t) for each strain s.

        MUST BE CONTINUOUS AND DIFFERENTIABLE FOR ALL TIMES t.

        Parameters
        ----------
        `t`: float as Traced<ShapedArray(float32[])>
            current time in the model, due to the just-in-time nature of Jax this float value may be contained within a
            traced array of shape () and size 1. Thus no explicit comparison should be done on "t".

        Returns
        -----------
        external_i_compartment: jnp.array()
            jnp.array(shape=(self.INITIAL_STATE[self.COMPARTMENT_IDX.I].shape)) of external individuals to the system
            interacting with susceptibles within the system, used to impact force of infection.
        """
        # set up our return value
        external_i_compartment = jnp.zeros(
            self.INITIAL_STATE[self.COMPARTMENT_IDX.I].shape
        )
        # default from the config
        external_i_distributions = self.EXTERNAL_I_DISTRIBUTIONS
        introduction_age_mask = jnp.where(
            jnp.array(self.INTRODUCTION_AGE_MASK),
            1,
            0,
        )
        for strain in self.STRAIN_IDX:
            external_i_distribution = external_i_distributions[strain]
            external_i_compartment = external_i_compartment.at[
                introduction_age_mask, 0, 0, strain
            ].set(
                external_i_distribution(t)
                * self.INTRODUCTION_PERCENTAGE
                * self.POPULATION[self.INTRODUCTION_AGE_MASK]
            )
        return external_i_compartment

    @partial(jax.jit, static_argnums=(0))
    def vaccination_rate(self, t):
        """
        Given some time t, returns a jnp.array of shape (self.NUM_AGE_GROUPS, self.MAX_VAX_COUNT + 1)
        representing the age / vax history stratified vaccination rates for an additional vaccine. Used by transmission models
        to determine vaccination rates at a particular time step.

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
            jnp.array(shape=(self.NUM_AGE_GROUPS, self.MAX_VAX_COUNT + 1)) of vaccination rates for each age bin and vax history strata.
        """
        return jnp.exp(
            utils.evaluate_cubic_spline(
                t,
                self.VAX_MODEL_KNOT_LOCATIONS,
                self.VAX_MODEL_BASE_EQUATIONS,
                self.VAX_MODEL_KNOTS,
            )
        )

    @partial(jax.jit, static_argnums=(0))
    def beta_coef(self, t):
        """Returns a coefficient for the beta value for cases of seasonal forcing or external impacts
        onto beta not direclty measured in the model. EG: masking mandates or holidays.
        Currently implemented via an array search with timings BETA_TIMES and coefficients BETA_COEFICIENTS

        Parameters
        ----------
        t: float as Traced<ShapedArray(float32[])>
            current time in the model, due to the just-in-time nature of Jax this float value may be contained within a
            traced array of shape () and size 1. Thus no explicit comparison should be done on "t".

        Returns:
        coefficient with which BETA can be multiplied with to externally increase or decrease the value to account for measures or seasonal forcing.
        """
        # this is basically a smart lookup function that works with JAX just in time compilation
        return self.BETA_COEFICIENTS[
            jnp.maximum(0, jnp.searchsorted(self.BETA_TIMES, t) - 1)
        ]

    def retrieve_population_counts(self):
        """
        A wrapper function which takes retrieves the age stratified population counts across all the INITIAL_STATE compartments
        (minus the book-keeping C compartment.)
        """
        self.POPULATION = np.sum(  # sum together S+E+I compartments
            np.array(
                [
                    np.sum(
                        compartment,
                        axis=(
                            self.S_AXIS_IDX.hist,
                            self.S_AXIS_IDX.vax,
                            self.S_AXIS_IDX.wane,
                        ),
                    )  # sum over all but age bin axis
                    for compartment in self.INITIAL_STATE[
                        : self.COMPARTMENT_IDX.C
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
        self.CROSSIMMUNITY_MATRIX:
            updates this matrix to shape (self.NUM_STRAINS, self.NUM_PREV_INF_HIST) containing the relative immune escape
            values for each challenging strain compared to each prior immune history in the model.
        """
        self.CROSSIMMUNITY_MATRIX = utils.strain_interaction_to_cross_immunity(
            self.NUM_STRAINS, self.STRAIN_INTERACTIONS
        )

    def load_vaccination_model(self):
        """
        loads parameters of a polynomial spline vaccination model stratified on age bin and current vaccination status.
        also loads in the spline knot locations.
        """
        parameters = pd.read_csv(self.VAX_MODEL_DATA)
        age_bins = len(parameters["age_group"].unique())
        vax_bins = len(parameters["dose"].unique())
        # change this if you start using higher degree polynomials to fit vax model
        assert age_bins == self.NUM_AGE_GROUPS, (
            "the number of age bins in your model does not match the input vaccination parameters, "
            + "please provide your own vaccination parameters that match, or adjust your age bins"
        )

        assert vax_bins == self.MAX_VAX_COUNT + 1, (
            "the number of vaccination counts in your model does not match the input vaccination parameters, "
            + "please provide your own vaccination parameters that match, or adjust your age bins"
        )
        vax_knots = np.zeros((age_bins, vax_bins, self.VAX_MODEL_NUM_KNOTS))
        vax_knot_locations = np.zeros(
            (age_bins, vax_bins, self.VAX_MODEL_NUM_KNOTS)
        )
        vax_base_equations = np.zeros((age_bins, vax_bins, 4))  # always 4
        for row in parameters.itertuples():
            _, age_group, vaccination = row[0:3]
            intersect_and_ts = row[3:7]
            knot_coefficients = row[7 : 7 + self.VAX_MODEL_NUM_KNOTS]
            knot_locations = row[7 + self.VAX_MODEL_NUM_KNOTS :]
            age_group_idx = self.AGE_GROUP_IDX[age_group]
            vax_idx = vaccination - 1
            vax_base_equations[age_group_idx, vax_idx, :] = np.array(
                intersect_and_ts
            )
            vax_knots[age_group_idx, vax_idx, :] = np.array(knot_coefficients)
            vax_knot_locations[age_group_idx, vax_idx, :] = np.array(
                knot_locations
            )
        self.VAX_MODEL_KNOTS = jnp.array(vax_knots)
        self.VAX_MODEL_KNOT_LOCATIONS = jnp.array(vax_knot_locations)
        self.VAX_MODEL_BASE_EQUATIONS = jnp.array(vax_base_equations)

    def load_external_i_distributions(self, introduction_times):
        """
        a function that loads external_i_distributions array into the model.
        this list of functions dictate the number of infected individuals EXTERNAL TO THE POPULATION are introduced at a particular timestep.

        each function within this list must be differentiable at all input values `t`>=0 and return some value such that
        sum(f(t)) forall t>=0 = 1.0. By default we use a normal PDF to approximate this value.

        Updates
        ----------
        EXTERNAL_I_DISTRIBUTIONS: list[func(jac_tracer(float))->float]
        updates each strain to have its own introduction function, centered around the corresponding introduction time in self.INTRODUCTION_TIMES
        historical strains, which are introduced before model initialization are given the zero function f(_) -> 0.
        """

        def zero_function(_):
            return 0

        self.EXTERNAL_I_DISTRIBUTIONS = [
            zero_function for _ in range(self.NUM_STRAINS)
        ]
        for introduced_strain_idx, introduced_time in enumerate(
            introduction_times
        ):
            # earlier introduced strains earlier will be placed closer to historical strains (0 and 1)
            dist_idx = (
                self.NUM_STRAINS
                - self.NUM_INTRODUCED_STRAINS
                + introduced_strain_idx
            )
            # use a normal PDF with std dv
            self.EXTERNAL_I_DISTRIBUTIONS[dist_idx] = partial(
                pdf, loc=introduced_time, scale=self.INTRODUCTION_SCALE
            )

    def load_contact_matrix(self):
        """
        a wrapper function that loads a contact matrix for the USA based on mixing paterns data found here:
        https://github.com/mobs-lab/mixing-patterns

        Updates
        ----------
        `self.CONTACT_MATRIX` : numpy.ndarray
            a matrix of shape (self.NUM_AGE_GROUPS, self.NUM_AGE_GROUPS) with each value representing TODO
        """
        self.CONTACT_MATRIX = utils.load_demographic_data(
            self.DEMOGRAPHIC_DATA_PATH,
            self.REGIONS,
            self.NUM_AGE_GROUPS,
            self.AGE_LIMITS[0],
            self.AGE_LIMITS,
        )["United States"]["avg_CM"]
