"""Inference utilities for DynODE NumPyro models.

This module is intentionally independent of the old ``dynode.config`` and
``dynode.typing`` APIs. It accepts any callable NumPyro model, including models
returned by ``DynodeModel.make_numpyro_model(...)`` and
``DynodeExperiment.make_numpyro_model(...)``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import arviz as az
import jax
from jax import Array
from jax.random import PRNGKey
from numpyro.infer import (
    MCMC,
    NUTS,
    SVI,
    Predictive,
    Trace_ELBO,
    init_to_median,
)
from numpyro.infer.autoguide import AutoContinuous, AutoMultivariateNormal
from numpyro.infer.hmc import HMCState
from numpyro.infer.svi import SVIRunResult
from numpyro.infer.util import log_likelihood
from numpyro.optim import Adam, _NumPyroOptim
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, PrivateAttr

NumpyroModel = Callable[..., Any]
PosteriorSamples = dict[str, Array]
PredictiveSamples = dict[str, Any]


class InferenceProcess(BaseModel):
    """Base class for fitting a NumPyro model.

    The model is any NumPyro-compatible callable. DynODE-specific objects such
    as ``DynodeModel`` or ``DynodeExperiment`` should be converted to a callable
    with ``make_numpyro_model(...)`` before being passed here.

    Examples
    --------
    Single model::

        model = dynode_model.make_numpyro_model()
        inferer = SVIProcess(numpyro_model=model, ...)
        inferer.infer(data=data)

    Multi-instance experiment::

        model = dynode_experiment.make_numpyro_model()
        inferer = SVIProcess(numpyro_model=model, ...)
        inferer.infer(data=data_by_instance)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    numpyro_model: NumpyroModel = Field(
        description=(
            "A NumPyro model callable. Usually this is returned by "
            "DynodeModel.make_numpyro_model(...) or "
            "DynodeExperiment.make_numpyro_model(...)."
        )
    )

    inference_prngkey: Array = Field(
        default_factory=lambda: PRNGKey(8675314),
        description="PRNG key used for inference and default predictive draws.",
    )

    _inference_complete: bool = PrivateAttr(default=False)
    _inferer: MCMC | SVI | None = PrivateAttr(default=None)
    _inference_state: HMCState | SVIRunResult | None = PrivateAttr(
        default=None
    )
    _inferer_kwargs: dict[str, Any] = PrivateAttr(default_factory=dict)

    @property
    def inference_complete(self) -> bool:
        """Whether ``infer(...)`` has completed successfully."""
        return self._inference_complete

    @property
    def inferer(self) -> MCMC | SVI | None:
        """Underlying NumPyro inferer, if inference has been run."""
        return self._inferer

    @property
    def inference_state(self) -> HMCState | SVIRunResult | None:
        """Final NumPyro inference state, if inference has been run."""
        return self._inference_state

    @property
    def inferer_kwargs(self) -> dict[str, Any]:
        """Model keyword arguments used for the most recent fit."""
        return dict(self._inferer_kwargs)

    def infer(self, **model_kwargs: Any) -> MCMC | SVI:
        """Fit ``numpyro_model``.

        Subclasses implement the specific inference algorithm. Keyword
        arguments are forwarded to ``numpyro_model``.
        """
        raise NotImplementedError(
            "Inference process not implemented; use MCMCProcess or SVIProcess."
        )

    def get_samples(
        self,
        group_by_chain: bool = False,
        exclude_deterministic: bool = True,
    ) -> PosteriorSamples:
        """Return posterior samples after inference."""
        raise NotImplementedError(
            "get_samples() not implemented; use MCMCProcess or SVIProcess."
        )

    def posterior_predictive(
        self,
        *,
        posterior_samples: Mapping[str, Any] | None = None,
        rng_key: Array | None = None,
        num_samples: int | None = None,
        return_sites: tuple[str, ...] | None = None,
        model_kwargs: Mapping[str, Any] | None = None,
    ) -> PredictiveSamples:
        """Generate posterior predictive samples.

        Parameters
        ----------
        posterior_samples:
            Samples to condition on. Defaults to ``self.get_samples()``.
        rng_key:
            PRNG key. Defaults to ``self.inference_prngkey``.
        num_samples:
            Optional number of samples for predictive draws. Usually omitted
            when ``posterior_samples`` are supplied.
        return_sites:
            Optional NumPyro return-site filter.
        model_kwargs:
            Optional model kwargs. Defaults to the kwargs from ``infer(...)``.
        """
        self._require_complete()

        samples = dict(posterior_samples or self.get_samples())
        kwargs = self._predictive_kwargs(model_kwargs)

        predictive = Predictive(
            self.numpyro_model,
            posterior_samples=samples,
            num_samples=num_samples,
            return_sites=return_sites,
        )

        return predictive(
            rng_key or self.inference_prngkey,
            **kwargs,
        )

    def prior_predictive(
        self,
        *,
        rng_key: Array | None = None,
        num_samples: int = 500,
        return_sites: tuple[str, ...] | None = None,
        model_kwargs: Mapping[str, Any] | None = None,
    ) -> PredictiveSamples:
        """Generate prior predictive samples from ``numpyro_model``."""
        kwargs = self._predictive_kwargs(model_kwargs)

        predictive = Predictive(
            self.numpyro_model,
            num_samples=num_samples,
            return_sites=return_sites,
        )

        return predictive(
            rng_key or self.inference_prngkey,
            **kwargs,
        )

    def log_likelihood(
        self,
        *,
        posterior_samples: Mapping[str, Any] | None = None,
        model_kwargs: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Compute pointwise log likelihood using NumPyro's utility."""
        self._require_complete()
        samples = dict(posterior_samples or self.get_samples())
        kwargs = self._predictive_kwargs(model_kwargs)
        return log_likelihood(self.numpyro_model, samples, **kwargs)

    def to_arviz(self) -> az.InferenceData:
        """Return results as an ArviZ ``InferenceData`` object."""
        raise NotImplementedError(
            "to_arviz() not implemented for the base InferenceProcess."
        )

    def _require_complete(self) -> None:
        if not self._inference_complete:
            raise AssertionError(
                "Inference process not completed; call infer(...) first."
            )

    def _predictive_kwargs(
        self,
        model_kwargs: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        if model_kwargs is None:
            return dict(self._inferer_kwargs)
        return dict(model_kwargs)


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


class SVIProcess(InferenceProcess):
    """Fit a NumPyro model with stochastic variational inference."""

    num_iterations: PositiveInt = Field(
        description="Number of SVI optimization steps."
    )
    num_samples: PositiveInt = Field(
        description="Number of approximate posterior samples returned by get_samples()."
    )

    guide_class: type[AutoContinuous] = AutoMultivariateNormal
    guide_init_strategy: Callable[..., Any] = init_to_median

    optimizer: _NumPyroOptim = Field(
        default_factory=lambda: Adam(step_size=0.1),
        description="NumPyro optimizer, for example Adam or ClippedAdam.",
    )

    progress_bar: bool = True
    guide_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra keyword arguments forwarded to the autoguide.",
    )

    def infer(self, **model_kwargs: Any) -> SVI:
        """Run SVI and store the fitted inferer/result."""
        guide = self.guide_class(
            self.numpyro_model,
            init_loc_fn=self.guide_init_strategy,
            **self.guide_kwargs,
        )

        inferer = SVI(
            model=self.numpyro_model,
            guide=guide,
            optim=self.optimizer,
            loss=Trace_ELBO(),
        )

        result = inferer.run(
            rng_key=self.inference_prngkey,
            num_steps=self.num_iterations,
            progress_bar=self.progress_bar,
            **model_kwargs,
        )

        self._inference_complete = True
        self._inferer = inferer
        self._inference_state = result
        self._inferer_kwargs = dict(model_kwargs)
        return inferer

    def get_samples(
        self,
        group_by_chain: bool = False,
        exclude_deterministic: bool = True,
    ) -> PosteriorSamples:
        """Return samples from the fitted variational posterior.

        ``group_by_chain`` is accepted for API compatibility. SVI has no chains,
        so the argument is ignored.
        """
        del group_by_chain
        self._require_complete()
        assert isinstance(self._inferer, SVI)
        assert isinstance(self._inference_state, SVIRunResult)

        predictive = Predictive(
            self._inferer.guide,
            params=self._inference_state.params,
            num_samples=self.num_samples,
        )
        samples = predictive(self.inference_prngkey)

        if not exclude_deterministic:
            deterministic_predictive = Predictive(
                model=self._inferer.model,
                guide=self._inferer.guide,
                params=self._inference_state.params,
                num_samples=self.num_samples,
            )
            deterministic_samples = deterministic_predictive(
                self.inference_prngkey,
                **self._inferer_kwargs,
            )
            samples = {**samples, **deterministic_samples}

        return _filter_internal_sites(samples)

    def posterior_predictive(
        self,
        *,
        posterior_samples: Mapping[str, Any] | None = None,
        rng_key: Array | None = None,
        num_samples: int | None = None,
        return_sites: tuple[str, ...] | None = None,
        model_kwargs: Mapping[str, Any] | None = None,
    ) -> PredictiveSamples:
        """Generate posterior predictive samples.

        For SVI, defaults to drawing from the fitted guide if explicit
        ``posterior_samples`` are not supplied.
        """
        self._require_complete()
        assert isinstance(self._inferer, SVI)
        assert isinstance(self._inference_state, SVIRunResult)

        kwargs = self._predictive_kwargs(model_kwargs)
        key = rng_key or self.inference_prngkey

        if posterior_samples is not None:
            predictive = Predictive(
                self.numpyro_model,
                posterior_samples=dict(posterior_samples),
                num_samples=num_samples,
                return_sites=return_sites,
            )
            return predictive(key, **kwargs)

        predictive = Predictive(
            model=self._inferer.model,
            guide=self._inferer.guide,
            params=self._inference_state.params,
            num_samples=num_samples or self.num_samples,
            return_sites=return_sites,
        )
        return predictive(key, **kwargs)

    def to_arviz(self) -> az.InferenceData:
        """Convert SVI outputs to ArviZ ``InferenceData``.

        ArviZ has no NumPyro SVI object equivalent to MCMC, so this method
        returns prior, posterior predictive, and log-likelihood groups.
        """
        self._require_complete()

        posterior_samples = self.get_samples()
        posterior_predictive = self.posterior_predictive(
            posterior_samples=posterior_samples,
        )
        prior = self.prior_predictive(num_samples=self.num_samples)
        ll = self.log_likelihood(posterior_samples=posterior_samples)

        return az.from_numpyro(
            prior=prior,
            posterior_predictive=posterior_predictive,
            log_likelihood=ll,
        )


def split_key(key: Array, num: int = 2) -> tuple[Array, ...]:
    """Small convenience wrapper around ``jax.random.split``."""
    return tuple(jax.random.split(key, num))


def _filter_internal_sites(samples: Mapping[str, Any]) -> PosteriorSamples:
    """Drop AutoGuide/private NumPyro sites from a samples dictionary."""
    return {
        name: value
        for name, value in samples.items()
        if not name.startswith("_auto_") and name != "auto_latent"
    }


__all__ = [
    "NumpyroModel",
    "PosteriorSamples",
    "PredictiveSamples",
    "InferenceProcess",
    "MCMCProcess",
    "SVIProcess",
    "split_key",
]
