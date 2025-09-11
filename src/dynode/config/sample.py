# """A utility module containing helper code for sampling and resolving Dynode objects."""
#
# from copy import deepcopy
#
## importing under a different name because mypy static type hinter
## strongly dislikes the IntEnum class.
# from typing import Any
#
# import numpy as np
# import numpyro  # type: ignore
# import numpyro.distributions as Dist  # type: ignore
# from jax import Array
# from pydantic import BaseModel
#
# from .deterministic_parameter import DeterministicParameter
# from .params import ParameterSet
#
#
# def sample_distributions(
#    obj: Any, rng_key: Array | None = None, _prefix: str = ""
# ):
#    """Recurisvely scans data structures and samples numpyro.Distribution objects.
#
#    Parameters
#    ----------
#    obj: Any
#        object to be sampled or searched for distributions.
#    rng_key : Array
#        optional rng_key to use if this function is called outside of an
#        mcmc or inference context where one is automatically provided.
#    _prefix : str
#        prefix to prepend to all site names, used to build up site names.
#
#    Note
#    ----
#    Sampled distributions receive site names according to a set of rules
#    - Distributions within lists are appended with _i identifing the index,
#    N times for N dimensional arrays
#    - Dictionaries and Pydnatic models are recursively searched with
#    sites names prepending the key that parameter belongs to.
#
#    Returns
#    -------
#    Any | jax.Array
#        obj with all instances of `numpyro.Distributions` sampled, if `obj`
#        itself is a distribution, the sample will be returned.
#    """
#    # if you wish to add custom search and prefix behavior for certain classes
#    # do so at the TOP of this if branch, so BaseModel does not catch your class.
#    if isinstance(obj, (BaseModel, dict)):
#        # Create dictionary representation if not and recursively sample fields.
#        obj_dict = dict(obj)
#        for key, value in obj_dict.items():
#            obj_dict[key] = sample_distributions(
#                value, rng_key=rng_key, _prefix=_prefix + f"{key}_"
#            )
#        return (
#            dict(obj_dict)
#            if isinstance(obj, dict)
#            else obj.__class__(**obj_dict)
#        )
#    elif isinstance(obj, (np.ndarray, list)):
#        # Recursively sample elements in list.
#        lst = [
#            sample_distributions(
#                item, rng_key=rng_key, _prefix=_prefix + f"{i}_"
#            )
#            for i, item in enumerate(obj)
#        ]
#
#        return lst
#    elif issubclass(type(obj), Dist.Distribution):
#        # Sample from distribution using JAX random key.
#        # remove trailing underscore from recursive calls above.
#        if len(_prefix) > 0:
#            _prefix = _prefix[:-1]
#        return numpyro.sample(_prefix, obj, rng_key=rng_key)
#    else:
#        # Return value as-is if not recognized type.
#        return obj
#
#
# def resolve_deterministic(
#    obj: Any, root_params: dict | BaseModel, _prefix: str = ""
# ):
#    """Find and resolve all DeterministicParameter types.
#
#    Parameters
#    ----------
#    obj: Any
#        Python data structure that may or may not contain DeterministicParameter
#        objects. If they exist will be resolved.
#    root_params: dict | Basemodel
#        dict or pydantic model used to resolve `DeterministicParameter`, all
#        parameters pointed to by DeterministicParameters must be in the
#        top level of this object.
#    _prefix:
#        optional prefix to add before site names, impacts downstream
#        inference functionality so best left alone.
#
#    Returns
#    -------
#    obj
#        The obj, with any `DeterministicParameter` object resolved within.
#        if `isinstance(obj, DeterministicParameter)` then the resolved value
#        is returned instead.
#
#    Examples
#    --------
#    >>> import numpyro.distributions as dist
#    ... from dynode.model_configuration.types import DeterministicParameter
#    ... from dynode.utils import sample_if_distribution, resolve_if_dependent
#    ... import numpyro.handlers as handlers
#
#    >>> parameters = {"x": dist.Normal(),
#    ...               "y": DeterministicParameter("x"),
#    ...               "x_lst": [0, dist.Normal(), 2],
#    ...               "y_lst": [0, DeterministicParameter("x_lst", index=1), 2]}
#
#    >>> with handlers.seed(rng_seed=1):
#    ...     samples = sample_distributions(parameters)
#    ...     resolved = resolve_deterministic(samples, root_params=samples)
#    >>> resolved
#        {'x': Array(-0.80760655, dtype=float64),
#        'y': Array(-0.80760655, dtype=float64),
#        'x_lst': Array([0.        , 0.57522288, 2.        ], dtype=float64),
#        'y_lst': Array([0.        , 0.57522288, 2.        ], dtype=float64)}
#    """
#    if isinstance(root_params, BaseModel):
#        root_params = dict(root_params)
#    # if you wish to add custom search and prefix behavior for certain classes
#    # do so at the TOP of this if branch, so BaseModel does not catch your class.
#    if isinstance(obj, (BaseModel, dict)):
#        # Create dictionary representation if not and recursively sample fields.
#        obj_dict = dict(obj)
#        for key, value in obj_dict.items():
#            obj_dict[key] = resolve_deterministic(
#                value, root_params, _prefix=_prefix + f"{key}_"
#            )
#        return (
#            dict(obj_dict)
#            if isinstance(obj, dict)
#            else obj.__class__(**obj_dict)
#        )
#    elif isinstance(obj, (np.ndarray, list)):
#        # Recursively sample elements in list.
#        return [
#            resolve_deterministic(item, root_params, _prefix=_prefix + f"{i}_")
#            for i, item in enumerate(obj)
#        ]
#    elif isinstance(obj, DeterministicParameter):
#        # resolve the DeterministicParameter by finding what its connected to
#        # remove trailing underscore from recursive calls above.
#        if len(_prefix) > 0:
#            _prefix = _prefix[:-1]
#        return numpyro.deterministic(_prefix, obj.resolve(root_params))
#    else:
#        # Return value as-is if not recognized type.
#        return obj
#
#
# def sample_then_resolve(
#    parameters: Any, rng_key: Array | None = None, _prefix: str = ""
# ) -> ParameterSet:
#    """Copy, sample and resolve parameters, returning a jax-compliant copy.
#
#    Parameters
#    ----------
#    parameters : Any
#        object containing numpyro.Distribution objects to sample and
#        dynode.typing.DeterministicParameter objects to resolve.
#
#    rng_key : Array, optional
#        PRNGKey needed to sample distributions, generated from
#        jax.random.PRNGKey(), by default None meaning context RNGKey will be
#        used if running from within MCMC execution.
#
#    _prefix : str, optional
#        prefix to append to all sampled and resolved parameters. Useful for
#        differentiating between different fits. Changing this parameter
#        may break code that depends on a hardcoded parameter name.
#        Defaults to "".
#
#    Returns
#    ---------
#    Any
#        COPY of the `parameters` object with all occurences of
#        `numpyro.Distribution` or `DeterministicParameter` objects replaced
#        with samples / resolved values.
#    """
#    parameters = deepcopy(parameters)
#    parameters = sample_distributions(
#        parameters, rng_key=rng_key, _prefix=_prefix
#    )
#    parameters = resolve_deterministic(
#        parameters, root_params=dict(parameters), _prefix=_prefix
#    )
#
#    return parameters
import dataclasses as dc
from typing import Any

import jax.tree_util as jtu
import numpyro
import numpyro.distributions as dist
import tree  # dm-tree
from pydantic import BaseModel

from .deterministic_parameter import DeterministicParameter
from .params import ParameterSet

# ---------- PyTree registrations (lazy, per concrete class) ----------


def _bm_flatten(x: BaseModel):
    # v2: model_dump(); v1: x.dict()
    d = x.model_dump()
    keys = tuple(sorted(d.keys()))
    children = [d[k] for k in keys]
    aux = (x.__class__, keys)
    return children, aux


def _bm_unflatten(aux, children):
    cls, keys = aux
    return cls(**{k: v for k, v in zip(keys, children)})


def _ps_flatten(x: ParameterSet):
    # If ParameterSet subclasses BaseModel, the BaseModel hook will handle it.
    d = dict(x) if not isinstance(x, BaseModel) else x.model_dump()
    keys = tuple(sorted(d.keys()))
    children = [d[k] for k in keys]
    aux = (x.__class__, keys)
    return children, aux


def _ps_unflatten(aux, children):
    cls, keys = aux
    return cls(**{k: v for k, v in zip(keys, children)})


def _dc_flatten(x: Any):
    # Generic dataclass support
    flds = dc.fields(x)
    keys = tuple(f.name for f in flds)
    children = [getattr(x, k) for k in keys]
    aux = (x.__class__, keys)
    return children, aux


def _dc_unflatten(aux, children):
    cls, keys = aux
    return cls(**{k: v for k, v in zip(keys, children)})


def _ensure_registered(x: Any):
    t = type(x)
    try:
        if isinstance(x, BaseModel):
            jtu.register_pytree_node(t, _bm_flatten, _bm_unflatten)
        elif isinstance(x, ParameterSet) and not isinstance(x, BaseModel):
            jtu.register_pytree_node(t, _ps_flatten, _ps_unflatten)
        elif dc.is_dataclass(x):
            jtu.register_pytree_node(t, _dc_flatten, _dc_unflatten)
    except ValueError:
        # already registered
        pass


def _register_walk(obj: Any):
    _ensure_registered(obj)
    if isinstance(obj, dict):
        for v in obj.values():
            _register_walk(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _register_walk(v)
    # jnp/np arrays remain leaves (good); nested containers are covered above


# ---------- Path → name ----------


def _path_to_name(path_entries, root_prefix: str | None) -> str:
    tokens: list[str] = []
    if root_prefix:
        tokens.append(root_prefix)
    for p in path_entries:
        if hasattr(p, "key"):
            tokens.append(str(p.key))  # dict key
        elif hasattr(p, "idx"):
            tokens.append(str(p.idx))  # sequence index
        else:
            tokens.append(str(p))
    return "_".join(tokens)


# ---------- Public API ----------


def sample_then_resolve(parameters: Any, root_prefix: str = "") -> Any:
    """
    Replace all numpyro Distribution leaves with samples and then
    resolve DeterministicParameter leaves. Returns a new structure.
    """
    # 0) make sure dm-tree can descend into your containers
    _register_walk(parameters)

    # 1) sample all distributions
    def _sample_leaf(path, x):
        if isinstance(x, dist.Distribution):
            name = _path_to_name(path, root_prefix or None)
            return numpyro.sample(name, x)
        return x

    sampled = tree.map_structure_with_path(_sample_leaf, parameters)

    # 2) resolve deterministics using the sampled structure
    def _resolve_leaf(path, x):
        if isinstance(x, DeterministicParameter):
            name = _path_to_name(path, root_prefix or None)
            return numpyro.deterministic(name, x.resolve(sampled))
        return x

    resolved = tree.map_structure_with_path(_resolve_leaf, sampled)

    # 3) (optional) fail fast if any Distributions slipped through
    # def _has_dist(z: Any) -> bool:
    #     leaves = tree.flatten(z)[0]
    #     return any(isinstance(l, dist.Distribution) for l in leaves)
    # assert not _has_dist(resolved), "Unsampled Distribution found after sample_then_resolve."

    return resolved
