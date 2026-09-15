from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import arviz as az
import numpy as np
from jax import Array
from numpyro.infer import (
    SVI,
    Predictive,
    Trace_ELBO,
    init_to_median,
)
from numpyro.infer.autoguide import AutoGuide, AutoMultivariateNormal
from numpyro.infer.svi import SVIRunResult
from numpyro.optim import Adam, _NumPyroOptim
from pydantic import Field, PositiveInt

from .inference_process import InferenceProcess

PosteriorSamples = dict[str, Array]
PredictiveSamples = dict[str, Any]


class SVIProcess(InferenceProcess):
    """Fit a NumPyro model with stochastic variational inference."""

    num_iterations: PositiveInt = Field(
        description="Number of SVI optimization steps."
    )
    num_samples: PositiveInt = Field(
        description="Number of approximate posterior samples returned by get_samples()."
    )

    guide_class: type[AutoGuide] = AutoMultivariateNormal
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

        print("Guide class:", self.guide_class)
        print("Guide kwargs:", self.guide_kwargs)
        print("Inference key:", self.inference_prngkey)
        guide = self.guide_class(
            self.numpyro_model,
            init_loc_fn=self.guide_init_strategy,
            **self.guide_kwargs,
        )

        print("Constructed guide:", guide)
        print(
            "Guide init scale:",
            getattr(
                guide,
                "init_scale",
                getattr(guide, "_init_scale", "not exposed"),
            ),
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

        losses = np.asarray(result.losses, dtype=float)

        print("Num losses:", len(losses))
        print("Finite losses:", np.isfinite(losses).sum())
        print("NaN losses:", np.isnan(losses).sum())
        print("Inf losses:", np.isinf(losses).sum())

        bad_idx = np.where(~np.isfinite(losses))[0]
        print("First bad loss index:", bad_idx[0] if len(bad_idx) else None)
        print("Last 20 losses:", losses[-20:])

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


def _filter_internal_sites(samples: Mapping[str, Any]) -> PosteriorSamples:
    """Drop AutoGuide/private NumPyro sites from a samples dictionary."""
    return {
        name: value
        for name, value in samples.items()
        if not name.startswith("_auto_") and name != "auto_latent"
    }
