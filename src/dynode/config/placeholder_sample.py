"""A Place-holder sample designed to error if sampled on its own."""

import numpyro.distributions as dist


class SamplePlaceholderError(Exception):
    """A special error raised if you attempt to randomly sample a placeholder variable."""

    pass


class PlaceholderSample(dist.Distribution):
    """A parameter that draws its values from an external set of samples."""

    def __init__(self):
        """Create a PlaceholderSample distribution."""
        super().__init__()

    def sample(self, _, sample_shape=()):
        """Retrieve sample from an external set of samples.

        Raises
        ------
        SamplePosteriorError
            if sample is called outside of an in-place substitute context like
            numpyro.handlers.substitute() or numpyro.infer.Predictive.
        """
        raise SamplePlaceholderError(
            "Attempted to sample a PosteriorSample parameter outside of a "
            "Predictive() context. This likely means you did not provide "
            "posterior samples to the context via numpyro.infer.Predictive() or "
            "numpyro.handlers.substitute()."
        )
