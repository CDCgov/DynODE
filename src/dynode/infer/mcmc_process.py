from __future__ import annotations

from collections.abc import Callable
from typing import Any

import arviz as az
from jax import Array
from numpyro.infer import (
    MCMC,
    NUTS,
    init_to_median,
)
from pydantic import Field, PositiveInt

from .inference_process import InferenceProcess

PosteriorSamples = dict[str, Array]


class MCMCProcess(InferenceProcess):
    """Fit a NumPyro model with NUTS/MCMC."""

    num_samples: PositiveInt
    num_warmup: PositiveInt
    num_chains: PositiveInt = 1
    nuts_max_tree_depth: PositiveInt = 10

    nuts_init_strategy: Callable[..., Any] = init_to_median

    mcmc_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra keyword arguments forwarded to numpyro.infer.MCMC.",
    )
    nuts_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra keyword arguments forwarded to numpyro.infer.NUTS.",
    )
    progress_bar: bool = True

    def infer(self, **model_kwargs: Any) -> MCMC:
        """Run NUTS/MCMC and store the fitted inferer."""
        kernel = NUTS(
            self.numpyro_model,
            dense_mass=True,
            max_tree_depth=self.nuts_max_tree_depth,
            init_strategy=self.nuts_init_strategy,
            **self.nuts_kwargs,
        )

        inferer = MCMC(
            kernel,
            num_warmup=self.num_warmup,
            num_samples=self.num_samples,
            num_chains=self.num_chains,
            progress_bar=self.progress_bar,
            **self.mcmc_kwargs,
        )

        inferer.run(
            rng_key=self.inference_prngkey,
            **model_kwargs,
        )

        self._inference_complete = True
        self._inferer = inferer
        self._inference_state = inferer.last_state
        self._inferer_kwargs = dict(model_kwargs)
        return inferer

    def get_samples(
        self,
        group_by_chain: bool = False,
        exclude_deterministic: bool = True,
    ) -> PosteriorSamples:
        """Return MCMC posterior samples."""
        self._require_complete()
        assert isinstance(self._inferer, MCMC)

        if exclude_deterministic:
            return self._inferer.get_samples(group_by_chain=group_by_chain)

        # NumPyro does not expose deterministic-site inclusion through a stable
        # public MCMC API on all supported versions. Preserve the old DynODE
        # behavior while localizing private-attribute usage here.
        sample_field = self._inferer._sample_field
        if group_by_chain:
            return self._inferer._states[sample_field]
        return self._inferer._states_flat[sample_field]

    def to_arviz(self) -> az.InferenceData:
        """Convert MCMC fit to ArviZ ``InferenceData``."""
        self._require_complete()
        assert isinstance(self._inferer, MCMC)

        posterior_predictive = self.posterior_predictive()
        prior = self.prior_predictive(num_samples=self.num_samples)

        return az.from_numpyro(
            self._inferer,
            prior=prior,
            posterior_predictive=posterior_predictive,
        )
