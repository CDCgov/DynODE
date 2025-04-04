"""Define available Dynode inference processes."""

from typing import Optional, Type

from diffrax import Solution
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
from numpyro.infer.svi import SVIRunResult
from numpyro.optim import Adam, _NumPyroOptim
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, PrivateAttr
from typing_extensions import Callable

from ..typing import ObservedData
from .config_definition import SimulationConfig


class InferenceProcess(BaseModel):
    """An Inference process for fitting a CompartmentalModel to data.

    Meant to be an Abstract class for specific inference methods.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    # TODO change this naming and the word model
    simulator: Callable[
        [SimulationConfig, Optional[ObservedData]], Solution
    ] = Field(
        description="""Model that initializes state, samples and resolves
        parameters, generates timeseries, and optionally compares it to
        observed data, returning generated data."""
    )
    inference_prngkey: Array = PRNGKey(8675314)
    _inference_complete: bool = PrivateAttr(default=False)
    _inferer: Optional[MCMC | SVI] = PrivateAttr(default=None)

    def infer(self, **kwargs) -> MCMC | SVI:
        """Fit the simulator to data using the inference process.

        Additional keyword arguments are passed to the simulator.

        Returns
        -------
        MCMC | SVI
            The MCMC or SVI object used for inference.
        """
        raise NotImplementedError(
            "Inference process not implemented, please use a subclass."
        )

    def get_samples(self) -> dict[str, Array]:
        """Get the posterior samples from the inference process.

        Returns
        -------
        dict[str, Array]
        A dictionary of posterior samples, where keys are parameter sites
        and values are the corresponding samples, possibly arranged by
        chain/sample in the case of MCMC.
        """
        raise NotImplementedError(
            "Inference process not implemented, please use a subclass."
        )


class MCMCProcess(InferenceProcess):
    """Inference process for fitting a simulator to data using MCMC."""

    num_samples: PositiveInt
    num_warmup: PositiveInt
    num_chains: PositiveInt
    nuts_max_tree_depth: PositiveInt
    nuts_init_strategy: Callable = init_to_median
    # TODO add docstring link to NUTS keyword docs
    mcmc_kwargs: dict = Field(
        default_factory=lambda: dict(),
        description="""Extra kwargs to MCMC, for more info see:
          https://num.pyro.ai/en/stable/mcmc.html""",
    )
    nuts_kwargs: dict = Field(
        default_factory=lambda: dict(),
        description="""Extra kwargs to NUTS sampler, for more info see:
        https://num.pyro.ai/en/latest/mcmc.html#numpyro.infer.hmc.NUTS""",
    )
    progress_bar: bool = True

    def infer(self, **kwargs) -> MCMC:
        """Fit the simulator to data using MCMC.

        Additional keyword arguments are passed to the simulator.

        Returns
        -------
        MCMC
            The MCMC object used for inference.
        """
        inferer = MCMC(
            NUTS(
                self.simulator,
                dense_mass=True,
                max_tree_depth=self.nuts_max_tree_depth,
                init_strategy=self.nuts_init_strategy,
                **self.nuts_kwargs,
            ),
            num_warmup=self.num_warmup,
            num_samples=self.num_samples,
            num_chains=self.num_chains,
            progress_bar=self.progress_bar,
            **self.mcmc_kwargs,
        )
        inferer.run(rng_key=self.inference_prngkey, **kwargs)
        self._inference_complete = True
        self._inferer = inferer
        return inferer

    def get_samples(self) -> dict[str, Array]:
        """Get the posterior samples from the inference process.

        Returns
        -------
        dict[str, Array]
        A dictionary of posterior samples, where keys are parameter sites
        and values are the corresponding samples, arranged with shape
        (num_chains, num_samples).
        """
        if not self._inference_complete:
            raise AssertionError(
                "Inference process not completed, please call infer() first."
            )
        assert isinstance(self._inferer, MCMC)
        return self._inferer.get_samples()


class SVIProcess(InferenceProcess):
    """Inference process for fitting a simulator to data using SVI."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    num_iterations: PositiveInt
    guide_class: Type[AutoContinuous] = AutoMultivariateNormal
    guide_init_strategy: Callable = init_to_median
    optimizer: _NumPyroOptim = Field(
        default_factory=lambda: Adam(step_size=0.1),
        description="""SVI optimizer, usually Adam, for available optimizers
        see: https://num.pyro.ai/en/stable/optimizers.html""",
    )
    progress_bar: bool = True
    guide_kwargs: dict = Field(
        default_factory=lambda: dict(),
        description="""extra kwargs to guide, for more information see:
        https://num.pyro.ai/en/stable/autoguide.html""",
    )
    _inferer_kwargs: Optional[dict] = PrivateAttr(
        default_factory=lambda: dict()
    )
    _inference_state: Optional[SVIRunResult] = PrivateAttr(default=None)

    def infer(self, **kwargs) -> SVI:
        """Fit the simulator to data using SVI.

        Additional keyword arguments are passed to the simulator.

        Returns
        -------
        SVI
            The SVI object used for inference.
        """
        guide = self.guide_class(
            self.simulator,
            init_loc_fn=self.guide_init_strategy,
            **self.guide_kwargs,
        )

        inferer = SVI(
            model=self.simulator,
            guide=guide,
            optim=self.optimizer,
            loss=Trace_ELBO(),
        )
        svi_state = inferer.init(self.inference_prngkey, **kwargs)
        self._inference_state = inferer.run(
            rng_key=self.inference_prngkey,
            num_steps=self.num_iterations,
            progress_bar=self.progress_bar,
            init_state=svi_state,
            **kwargs,
        )
        self._inference_complete = True
        self._inferer = inferer
        self._inferer_kwargs = kwargs
        return inferer

    def get_samples(self) -> dict[str, Array]:
        """Get the posterior samples from the inference process.

        Returns
        -------
        dict[str, Array]
            A dictionary of posterior samples, where keys are parameter sites
            and values are the corresponding samples.

        Notes
        -----
        Keep in mind that posterior samples are generated after the fitting
        process for SVI, and the samples are not arranged by chain/sample like in MCMC.
        """
        if not self._inference_complete:
            raise AssertionError(
                "Inference process not completed, please call infer() first."
            )
        assert isinstance(self._inference_state, SVIRunResult)
        assert isinstance(self._inferer, SVI)

        # Construct the variational posterior distribution
        predictive = Predictive(
            self._inferer.guide,
            params=self._inference_state.params,
            num_samples=self.num_iterations,
        )
        samples = predictive(self.inference_prngkey)
        deterministic_predictive = Predictive(
            model=self._inferer.model,
            guide=self._inferer.guide,
            params=self._inference_state.params,
            num_samples=self.num_iterations,
        )
        # TODO revist this, is this what we want to be doing with rng here?
        rng_key_deterministic = self.inference_prngkey + 1

        deterministic_samples = deterministic_predictive(
            rng_key_deterministic, **self._inferer_kwargs
        )

        # Filter out internal parameters (like auto_latent)
        filtered_samples = {
            name: value
            for name, value in samples.items()
            if not name.startswith("_auto_")
        }

        combined_samples = {
            **deterministic_samples,
            **filtered_samples,
        }
        # TODO, decide if we want a filtering process for deterministic samples.
        return combined_samples
