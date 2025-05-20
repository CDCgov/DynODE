"""utility functions used within various components of initialization, inference, and interpretation."""

from typing import Any, Callable, List

import numpy as np
import pandas as pd  # type: ignore
from jax import Array

pd.options.mode.chained_assignment = None


def vectorize_objects(
    objs: List[Any],
    target: str,
    filter: Callable[[Any], bool] = lambda _: True,
) -> List[Any]:
    """Given a list of objects, return a list of the target attribute for each object, optionally applying a filter.

    Parameters
    ----------
    objs : List[obj]
        list of objects
    target : str
        attribute name to pull from each object in objs.
    filter : Callable[[obj], bool], optional
        function to filter the objects in objs, by default lambda _: True.
        if `filter(obj)` is False, object.target is not evaluated and excluded from returned list.

    Returns
    -------
    list[Any]
        Target attribute from each obj in objs such that filter(obj) is True.

    Raises
    ------
    KeyError
        if target is not a attribute of the obj and filter(obj) is True.
    """
    assert isinstance(target, str), "target must be a string"
    return [getattr(obj, target) for obj in objs if filter(obj)]


def flatten_list_parameters(
    samples: dict[str, np.ndarray | Array],
) -> dict[str, np.ndarray | Array]:
    """
    Flatten plated parameters into separate keys in the samples dictionary.

    Parameters
    ----------
    samples : dict[str, np.ndarray | Array]
        Dictionary with parameter names as keys and sample
        arrays as values. Arrays may have shape MxNxP for P independent draws.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary with plated parameters split into
        separate keys. Each new key has arrays of shape MxN.

    Notes
    -----
    If no plated parameters are present, returns a copy of the dictionary.
    """
    return_dict = {}
    for key, value in samples.items():
        if isinstance(value, (np.ndarray, Array)) and value.ndim > 2:
            num_dims = value.ndim - 2
            indices = (
                np.indices(value.shape[-num_dims:]).reshape(num_dims, -1).T
            )

            for idx in indices:
                new_key = f"{key}"
                for i in range(len(idx)):
                    new_key += f"_{idx[i]}"

                new_value = value[
                    tuple([slice(None)] * (value.ndim - num_dims) + list(idx))
                ]
                return_dict[new_key] = new_value
        else:
            return_dict[key] = value
    return return_dict


def drop_keys_with_substring(dct: dict[str, Any], drop_s: str):
    """
    Drop keys from a dictionary if they contain a specified substring.

    Parameters
    ----------
    dct : dict[str, Any]
        Dictionary with string keys.
    drop_s : str
        Substring to check for in keys.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys containing `drop_s` removed.
    """
    keys_to_drop = [key for key in dct.keys() if drop_s in key]
    for key in keys_to_drop:
        del dct[key]
    return dct
