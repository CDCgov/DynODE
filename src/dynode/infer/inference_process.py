from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import arviz as az
from jax import Array
from jax.random import PRNGKey
from numpyro.infer import (
    MCMC,
    SVI,
    Predictive,
)
from numpyro.infer.hmc import HMCState
from numpyro.infer.svi import SVIRunResult
from numpyro.infer.util import log_likelihood
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

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
