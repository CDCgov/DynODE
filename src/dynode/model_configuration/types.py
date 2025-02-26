"""Module for declaring types to be used within DynODE config files."""

from typing import Any, Optional

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


class DependentParameter:
    """A parameter whose value depends on a different parameter's value."""

    def __init__(self, depends_on: str, index: Optional[int | tuple] = None):
        """Link this DependentParameter to another parameter, possibly within a list"""
        self.depends_on = depends_on
        self.index = index

    def resolve(self, parameter_state: dict[str, Any]) -> Any:
        """Retrieve value in its current state. Marking it as deterministic

        Parameters
        ----------
        parameter_state : dict[str, Any]
            current parameters, must include `self.depends_on` in keys.

        Returns
        -------
        Any
            value at parameter_state[self.depends_on][self.index]

        Raises
        ------
        IndexError
            if parameter_state[self.depends_on][self.index] does not exist
        """
        if self.index is None:
            return parameter_state[self.depends_on]
        else:
            return parameter_state[self.depends_on][self.index]
