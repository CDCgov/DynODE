"""A utility module containing helper code for sampling and resolving Dynode objects."""

from copy import deepcopy

# importing under a different name because mypy static type hinter
# strongly dislikes the IntEnum class.
from typing import Any

import numpy as np
import numpyro  # type: ignore
import numpyro.distributions as Dist  # type: ignore
from jax import Array
from pydantic import BaseModel

from .model_configuration.params import TransmissionParams
from .typing import DeterministicParameter


def sample_distributions(obj: Any, rng_key: Array = None, _prefix: str = ""):
    """Recurisvely scans data structures and samples numpyro.Distribution objects.

    Parameters
    ----------
    obj: Any
        object to be sampled or searched for distributions.
    rng_key : Array
        optional rng_key to use if this function is called outside of an
        mcmc or inference context where one is automatically provided.
    _prefix : str
        prefix to prepend to all site names, used to build up site names.

    Note
    ----
    Sampled distributions receive site names according to a set of rules
    - Distributions within lists are appended with _i identifing the index,
    N times for N dimensional arrays
    - Dictionaries and Pydnatic models are recursively searched with
    sites names prepending the key that parameter belongs to.

    Returns
    -------
    Any | jax.Array
        obj with all instances of `numpyro.Distributions` sampled, if `obj`
        itself is a distribution, the sample will be returned.
    """
    # if you wish to add custom search and prefix behavior for certain classes
    # do so at the TOP of this if branch, so BaseModel does not catch your class.
    if isinstance(obj, (BaseModel, dict)):
        # Create dictionary representation if not and recursively sample fields.
        obj_dict = dict(obj)
        for key, value in obj_dict.items():
            obj_dict[key] = sample_distributions(
                value, rng_key=rng_key, _prefix=_prefix + f"{key}_"
            )
        return (
            dict(obj_dict)
            if isinstance(obj, dict)
            else obj.__class__(**obj_dict)
        )
    elif isinstance(obj, (np.ndarray, list)):
        # Recursively sample elements in list.
        lst = [
            sample_distributions(
                item, rng_key=rng_key, _prefix=_prefix + f"{i}_"
            )
            for i, item in enumerate(obj)
        ]

        return lst
    elif issubclass(type(obj), Dist.Distribution):
        # Sample from distribution using JAX random key.
        # remove trailing underscore from recursive calls above.
        if len(_prefix) > 0:
            _prefix = _prefix[:-1]
        return numpyro.sample(_prefix, obj, rng_key=rng_key)
    else:
        # Return value as-is if not recognized type.
        return obj


def resolve_deterministic(
    obj: Any, root_params: dict | BaseModel, _prefix: str = ""
):
    """Find and resolve all DeterministicParameter types.

    Parameters
    ----------
    obj: Any
        Python data structure that may or may not contain DeterministicParameter
        objects. If they exist will be resolved.
    root_params: dict | Basemodel
        dict or pydantic model used to resolve `DeterministicParameter`, all
        parameters pointed to by DeterministicParameters must be in the
        top level of this object.
    _prefix:
        optional prefix to add before site names, impacts downstream
        inference functionality so best left alone.

    Returns
    -------
    obj
        The obj, with any `DeterministicParameter` object resolved within.
        if `isinstance(obj, DeterministicParameter)` then the resolved value
        is returned instead.

    Examples
    --------
    >>> import numpyro.distributions as dist
    ... from dynode.model_configuration.types import DeterministicParameter
    ... from dynode.utils import sample_if_distribution, resolve_if_dependent
    ... import numpyro.handlers as handlers

    >>> parameters = {"x": dist.Normal(),
    ...               "y": DeterministicParameter("x"),
    ...               "x_lst": [0, dist.Normal(), 2],
    ...               "y_lst": [0, DeterministicParameter("x_lst", index=1), 2]}

    >>> with handlers.seed(rng_seed=1):
    ...     samples = sample_distributions(parameters)
    ...     resolved = resolve_deterministic(samples, root_params=samples)
    >>> resolved
        {'x': Array(-0.80760655, dtype=float64),
        'y': Array(-0.80760655, dtype=float64),
        'x_lst': Array([0.        , 0.57522288, 2.        ], dtype=float64),
        'y_lst': Array([0.        , 0.57522288, 2.        ], dtype=float64)}
    """
    if isinstance(root_params, BaseModel):
        root_params = dict(root_params)
    # if you wish to add custom search and prefix behavior for certain classes
    # do so at the TOP of this if branch, so BaseModel does not catch your class.
    if isinstance(obj, (BaseModel, dict)):
        # Create dictionary representation if not and recursively sample fields.
        obj_dict = dict(obj)
        for key, value in obj_dict.items():
            obj_dict[key] = resolve_deterministic(
                value, root_params, _prefix=_prefix + f"{key}_"
            )
        return (
            dict(obj_dict)
            if isinstance(obj, dict)
            else obj.__class__(**obj_dict)
        )
    elif isinstance(obj, (np.ndarray, list)):
        # Recursively sample elements in list.
        return [
            resolve_deterministic(item, root_params, _prefix=_prefix + f"{i}_")
            for i, item in enumerate(obj)
        ]
    elif isinstance(obj, DeterministicParameter):
        # resolve the DeterministicParameter by finding what its connected to
        # remove trailing underscore from recursive calls above.
        if len(_prefix) > 0:
            _prefix = _prefix[:-1]
        return numpyro.deterministic(_prefix, obj.resolve(root_params))
    else:
        # Return value as-is if not recognized type.
        return obj


def sample_then_resolve(
    parameters: Any, rng_key: Array = None
) -> TransmissionParams:
    """Copies, samples and resolves parameters, returning a jax-compliant copy.

    Parameters
    ----------
    parameters : Any
        object containing numpyro.Distribution objects to sample and
        dynode.typing.DeterministicParameter objects to resolve.

    rng_key : Array, optional
        PRNGKey needed to sample distributions, generated from
        jax.random.PRNGKey(), by default None meaning context RNGKey will be
        used if running from within MCMC execution.

    Returns
    ---------
    Any
        COPY of the `parameters` object with all occurences of
        `numpyro.Distribution` or `DeterministicParameter` objects replaced
        with samples / resolved values.
    """
    parameters = deepcopy(parameters)
    parameters = sample_distributions(parameters, rng_key=rng_key)
    parameters = resolve_deterministic(
        parameters, root_params=dict(parameters)
    )
    return parameters


def identify_distribution_indexes(
    parameters: dict[str, Any],
) -> dict[str, dict[str, str | tuple | None]]:
    """Identify the locations and site names of numpyro samples.

    The inverse of `sample_if_distribution()`, identifies which parameters
    are numpyro distributions and returns a mapping between the sample site
    names and its actual parameter name and index.

    Parameters
    ----------
    parameters : dict[str, Any]
        A dictionary containing keys of different parameter
        names and values of any type.

    Returns
    -------
    dict[str, dict[str, str | tuple[int] | None]]
        A dictionary mapping the sample name to the dict key within `parameters`.
        If the sampled parameter is within a larger list, returns a tuple of indexes as well,
        otherwise None.

        - key: `str`
            Sampled parameter name as produced by `sample_if_distribution()`.
        - value: `dict[str, str | tuple | None]`
            "sample_name" maps to key within `parameters` and "sample_idx" provides
            the indexes of the distribution if it is found in a list, otherwise None.

    Examples
    --------
    >>> import numpyro.distributions as dist
    >>> parameters = {"test": [0, dist.Normal(), 2], "example": dist.Normal()}
    >>> identify_distribution_indexes(parameters)
    {'test_1': {'sample_name': 'test', 'sample_idx': (1,)},
    'example': {'sample_name': 'example', 'sample_idx': None}}
    """

    def get_index(indexes):
        return tuple(indexes)

    index_locations: dict[str, dict[str, str | tuple | None]] = {}
    for key, param in parameters.items():
        # if distribution, it does not have an index, so None
        if issubclass(type(param), Dist.Distribution):
            index_locations[key] = {"sample_name": key, "sample_idx": None}
        # if list, check for distributions within and mark their indexes
        elif isinstance(param, (np.ndarray, list)):
            param = np.array(param)  # cast np.array so we get .shape
            flat_param = np.ravel(param)  # Flatten the parameter array
            # check for distributions inside of the flattened parameter list
            if any(
                [
                    issubclass(type(param_lst), Dist.Distribution)
                    for param_lst in flat_param
                ]
            ):
                dim_idxs = np.unravel_index(
                    np.arange(flat_param.size), param.shape
                )
                for i, param_lst in enumerate(flat_param):
                    if issubclass(type(param_lst), Dist.Distribution):
                        param_idxs = [dim_idx[i] for dim_idx in dim_idxs]
                        index_locations[
                            str(
                                key
                                + "_"
                                + "_".join(
                                    [str(dim_idx[i]) for dim_idx in dim_idxs]
                                )
                            )
                        ] = {
                            "sample_name": key,
                            "sample_idx": get_index(param_idxs),
                        }
    return index_locations
