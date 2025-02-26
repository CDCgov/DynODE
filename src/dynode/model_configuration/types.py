"""Module for declaring types to be used within DynODE config files."""

import numpyro.distributions as dist


class SamplePosteriorError(Exception):
    """A special error raised if you attempt to randomly sample a deterministic posterior draw."""

    pass


class PosteriorSample(dist.Distribution):
    """A parameter that draws its values from an external set of posterior samples."""

    def __init__(self):
        """Create a placeholder PosteriorSample distribution."""
        super().__init__()

    def sample(self, _, sample_shape=()):
        """Retrieve sample from a Posterior distribution.

        Raises
        ------
        SamplePosteriorError
            if sample is called outside of an in-place substitute context like
            numpyro.handlers.substitute() or numpyro.infer.Predictive.
        """
        raise SamplePosteriorError(
            "Attempted to sample a PosteriorSample parameter outside of a "
            "Predictive() context. This likely means you did not provide "
            "posterior samples to the context via numpyro.infer.Predictive() or "
            "numpyro.handlers.substitute()."
        )
