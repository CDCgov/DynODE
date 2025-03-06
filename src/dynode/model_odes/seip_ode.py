"""SEIP specific ODE class."""

import jax
import jax.numpy as jnp

from dynode.model_configuration.odes import ODEBase
from dynode.model_configuration.pre_packaged.covid_seip_config import (
    ODEParametersSEIP,
    SEIPCovidModel,
)
from dynode.typing import CompartmentGradients, SEIC_Compartments
from dynode.utils import get_foi_suscept, new_immune_state


class SEIP_COVID_ODE(ODEBase):
    """SEIP specific ODE class to solve covid SEIP models."""

    def __init__(self, compartmental_model: SEIPCovidModel):
        """Initialize an ODE specialized for SEIP covid models.

        Parameters
        ----------
        compartmental_model : SEIPCovidModel
            SEIP model that will be calling this ODE.
        """
        super().__init__(compartmental_model)
        pass

    def __call__(
        self,
        compartments: SEIC_Compartments,  # type: ignore[override]
        t: float,
        p: ODEParametersSEIP,  # type: ignore[override]
    ) -> CompartmentGradients:
        """Set of flows defining a SEIP (Susceptible, Exposed, Infectious, Partial) ODE model.

        In practice, S and P compartments are both defined within S, and the
        fourth compartment is instead used to track cumulative incidence (C)

        Parameters
        ----------
        state : pytree
            a tuple or any array-like object capable of unpacking, holding the current
            state of the model. In this case holding population values of the
            S, E, I, and C compartments.

        t : ArrayLike
            current time of the model in days

        parameters : ODEParametersSEIP
            parameters needed by the SEIP ODE model.

        Returns
        -------
        tuple[jax.Array, jax.Array, jax.Array, jax.Array]
            A tuple containing the rates of change of all compartments given in
            the `state` parameter. Each element in the return tuple will match the
            dimensions of the parallel element in `state`.

        Note
        ----
        Fully suseptible people are a subset of the suseptible compartment.

        Lots of Jax optimization done here as this is the most run bit of code in
        all of DynODE.
        """
        s, e, i, c = compartments

        ds, de, di, dc = (
            jnp.zeros(s.shape),
            jnp.zeros(e.shape),
            jnp.zeros(i.shape),
            jnp.zeros(c.shape),
        )
        # CALCULATING SUCCESSFULL INFECTIONS OF (partially) SUSCEPTIBLE INDIVIDUALS
        # including externally infected individuals to introduce new strains
        force_of_infection = (
            (
                p.beta
                * p.beta_coef(t)
                * p.seasonality(t)
                * jnp.einsum(
                    "ab,bijk->ak",
                    p.contact_matrix,
                    i + p.external_i(t),
                )
            ).transpose()
            / p.population
        ).transpose()  # (NUM_AGE_GROUPS, strain)

        foi_suscept = jnp.array(get_foi_suscept(p, force_of_infection))
        # we are vmaping this for loop. We select the force of infection
        # for each strain, and calculated the number of susceptibles it exposes
        # we sum over wane bin since `e` has no waning bin.
        # OLD FOR LOOP FOR INTERPRETABILITY
        # for strain in range(p.NUM_STRAINS):
        #     exposed_s = s * foi_suscept[strain]
        #     de = de.at[:, :, :, strain].add(
        #         jnp.sum(exposed_s, axis=-1)
        #     )
        #     ds = jnp.add(ds, -exposed_s)
        exposed_s = jnp.moveaxis(
            jax.vmap(
                lambda s, foi_suscept: s * foi_suscept,
                in_axes=(None, 0),
            )(s, foi_suscept),
            0,
            -1,
        )  # returns shape (s.shape..., p.NUM_STRAINS)
        # s has waning as last dimension, e has infected strain as last dim
        # the last two dimensions of `exposed_s` are `wane` and `strain`
        # so lets sum over them to get the expected shape for each
        de = de + jnp.sum(exposed_s, axis=-2)  # remove wane so matches e.shape
        ds = ds - jnp.sum(
            exposed_s, axis=-1
        )  # remove strain so matches s.shape
        dc = de  # at this point we only have infections in de, so we add to cumulative
        # e and i shape remain same, just multiplying by a constant.
        de_to_i = p.sigma * e  # exposure -> infectious
        di_to_w0 = p.gamma * i  # infectious -> new_immune_state
        di = jnp.add(de_to_i, -di_to_w0)
        de = jnp.add(de, -de_to_i)

        # go through all combinations of immune history and exposing strain
        # calculate new immune history after recovery, place them there.
        # THIS CODE REPLACES THE FOLLOWING FOR LOOP
        # for strain, immune_state in product(
        #     range(p.NUM_STRAINS), range(2**p.NUM_STRAINS)
        # ):
        #     new_state = new_immune_state(immune_state, strain, p.NUM_STRAINS)
        #     # recovered i->w0 transfer from `immune_state` -> `new_state` due to recovery from `strain`
        #     ds = ds.at[:, new_state, :, 0].add(
        #         di_to_w0[:, immune_state, :, strain]
        #     )
        def compute_ds(strain, immune_state, ds, di_to_w0):
            # Compute the updated values for ds for a single combination of strain and immune state
            # will be vectorized
            new_state = new_immune_state(immune_state, strain)
            # move them there
            recovered_individuals = (
                jnp.zeros(ds.shape)
                .at[:, new_state, :, 0]
                .add(di_to_w0[:, immune_state, :, strain])
            )
            return recovered_individuals

        # get all combinations of strain x immune history, jax version of cartesian product
        combinations = jnp.stack(
            jnp.meshgrid(
                jnp.arange(p.num_strains), jnp.arange(2**p.num_strains)
            ),
            axis=-1,
        ).reshape(-1, 2)
        assert len(combinations.T) == 2
        # compute vectorized function on all possible immune_hist x exposing strain
        ds_recovered = jnp.sum(
            jax.vmap(compute_ds, in_axes=(0, 0, None, None))(
                # Destructuring to tell mypy
                combinations.T[0],
                combinations.T[1],
                ds,
                di_to_w0,
            ),
            axis=0,
        )
        ds = ds + ds_recovered
        # lets measure our waned + vax rates
        # last w group doesn't wane but WANING_RATES enforces a 0 at the end
        waning_array = jnp.zeros(s.shape).at[:, :, :].add(p.waning_rates)
        s_waned = waning_array * s
        ds = ds.at[:, :, :, 1:].add(s_waned[:, :, :, :-1])
        ds = ds.at[:, :, :, :-1].add(-s_waned[:, :, :, :-1])

        # slice across age, strain, and wane. vaccination updates the vax column and also moves all to w0.
        # ex: diagonal movement from 1 shot in 4th waning compartment to 2 shots 0 waning compartment      s[:, 0, 1, 3] -> s[:, 0, 2, 0]
        # input vaccination rate is per entire population, need to update to per compartments first
        vax_rates = p.vaccination_rates(t)
        vax_totals = vax_rates * p.population[:, None]
        vax_status_counts = jnp.sum(
            s, axis=(1, 3)
        )  # Sum over immune hist and waning to get count per age and vax status
        updated_vax_rates = vax_totals / vax_status_counts
        updated_vax_rates = jnp.where(
            updated_vax_rates > 1.0,
            jnp.ones(updated_vax_rates.shape),
            updated_vax_rates,
        )  # prevent moving more people out than the compartments have

        # Assuming that people who received 2 or more doses wouldn't get additional booster too soon
        # i.e., when they were still within the first waning compartment
        vax_counts = s * updated_vax_rates[:, jnp.newaxis, :, jnp.newaxis]
        vax_counts = vax_counts.at[:, :, p.max_vaccination_count, 0].set(0)
        vax_gained = jnp.sum(vax_counts, axis=(-1))
        ds = ds.at[:, :, p.max_vaccination_count, 0].add(
            vax_gained[:, :, p.max_vaccination_count]
        )
        ds = ds.at[:, :, 1 : (p.max_vaccination_count) + 1, 0].add(
            vax_gained[:, :, 0 : p.max_vaccination_count]
        )
        ds = ds - vax_counts

        # if we are not implementing seasonal vaccination p.SEASONAL_VACCINATION_RESET(t) = 0 forall t
        # and you can safely ignore this section
        seasonal_vaccination_outflow = p.seasonal_vaccination_reset(t)
        # flow seasonal_vaccination_outflow% of seasonal vaxers back to max ordinal tier
        ds = ds.at[:, :, p.max_vaccination_count - 1, :].add(
            seasonal_vaccination_outflow * s[:, :, p.max_vaccination_count, :]
        )
        # remove these people from the seasonal vaccination tier
        ds = ds.at[:, :, p.max_vaccination_count, :].add(
            -seasonal_vaccination_outflow * s[:, :, p.max_vaccination_count, :]
        )
        # do the same process for e and i compartments
        de = de.at[:, :, p.max_vaccination_count - 1, :].add(
            seasonal_vaccination_outflow * e[:, :, p.max_vaccination_count, :]
        )
        de = de.at[:, :, p.max_vaccination_count, :].add(
            -seasonal_vaccination_outflow * e[:, :, p.max_vaccination_count, :]
        )
        di = di.at[:, :, p.max_vaccination_count - 1, :].add(
            seasonal_vaccination_outflow * i[:, :, p.max_vaccination_count, :]
        )
        di = di.at[:, :, p.max_vaccination_count, :].add(
            -seasonal_vaccination_outflow * i[:, :, p.max_vaccination_count, :]
        )

        return (ds, de, di, dc)
