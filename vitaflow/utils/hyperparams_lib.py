# Copyright 2018 The vitaFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Miscellaneous Utility functions.
"""

import copy
import collections
import numpy as np
import json

import tensorflow as tf
from vitaflow.internal.data_generators import problem as problem_lib

from vitaflow.utils import registry
from vitaflow.utils.hyperparams import HParams


def dict_patch(tgt_dict, src_dict):
    """Recursively patch :attr:`tgt_dict` by adding items from :attr:`src_dict`
    that do not exist in :attr:`tgt_dict`.

    If respective items in :attr:`src_dict` and :attr:`tgt_dict` are both
    `dict`, the :attr:`tgt_dict` item is patched recursively.

    Args:
        tgt_dict (dict): Target dictionary to patch.
        src_dict (dict): Source dictionary.

    Return:
        dict: The new :attr:`tgt_dict` that is patched.
    """
    if src_dict is None:
        return tgt_dict

    for key, value in src_dict.items():
        if key not in tgt_dict:
            tgt_dict[key] = copy.deepcopy(value)
        elif isinstance(value, dict) and isinstance(tgt_dict[key], dict):
            tgt_dict[key] = dict_patch(tgt_dict[key], value)
    return tgt_dict


def dict_lookup(dict_, keys, default=None):
    """Looks up :attr:`keys` in the dict, returns the corresponding values.

    The :attr:`default` is used for keys not present in the dict.

    Args:
        dict_ (dict): A dictionary for lookup.
        keys: A numpy array or a (possibly nested) list of keys.
        default (optional): Value to be returned when a key is not in
            :attr:`dict_`. Error is raised if :attr:`default` is not given and
            key is not in the dict.

    Returns:
        A numpy array of values with the same structure as :attr:`keys`.

    Raises:
        TypeError: If key is not in :attr:`dict_` and :attr:`default` is `None`.
    """
    return np.vectorize(lambda x: dict_.get(x, default))(keys)


def dict_fetch(src_dict, tgt_dict_or_keys):
    """Fetches a sub dict of :attr:`src_dict` with the keys in
    :attr:`tgt_dict_or_keys`.

    Args:
        src_dict: A dict or instance of :class:`~texar.HParams`.
            The source dict to fetch values from.
        tgt_dict_or_keys: A dict, instance of :class:`~texar.HParams`,
            or a list (or a dict_keys) of keys to be included in the output
            dict.

    Returns:
        A new dict that is a subdict of :attr:`src_dict`.
    """
    if src_dict is None:
        return src_dict

    if isinstance(tgt_dict_or_keys, HParams):
        tgt_dict_or_keys = tgt_dict_or_keys.todict()
    if isinstance(tgt_dict_or_keys, dict):
        tgt_dict_or_keys = tgt_dict_or_keys.keys()
    keys = list(tgt_dict_or_keys)

    if isinstance(src_dict, HParams):
        src_dict = src_dict.todict()

    return {k: src_dict[k] for k in keys if k in src_dict}


def dict_pop(dict_, pop_keys, default=None):
    """Removes keys from a dict and returns their values.

    Args:
        dict_ (dict): A dictionary from which items are removed.
        pop_keys: A key or a list of keys to remove and return respective
            values or :attr:`default`.
        default (optional): Value to be returned when a key is not in
            :attr:`dict_`. The default value is `None`.

    Returns:
        A `dict` of the items removed from :attr:`dict_`.
    """
    if not isinstance(pop_keys, (list, tuple)):
        pop_keys = [pop_keys]
    ret_dict = {key: dict_.pop(key, default) for key in pop_keys}
    return ret_dict


def flatten_dict(dict_, parent_key="", sep="."):
    """Flattens a nested dictionary. Namedtuples within the dictionary are
    converted to dicts.

    Adapted from:
    https://github.com/google/seq2seq/blob/master/seq2seq/models/model_base.py

    Args:
        dict_ (dict): The dictionary to flatten.
        parent_key (str): A prefix to prepend to each key.
        sep (str): Separator that intervenes between parent and child keys.
            E.g., if `sep` == '.', then `{ "a": { "b": 3 } }` is converted
            into `{ "a.b": 3 }`.

    Returns:
        A new flattened `dict`.
    """
    items = []
    for key, value in dict_.items():
        key_ = parent_key + sep + key if parent_key else key
        if isinstance(value, collections.MutableMapping):
            items.extend(flatten_dict(value, key_, sep=sep).items())
        elif isinstance(value, tuple) and hasattr(value, "_asdict"):
            dict_items = collections.OrderedDict(zip(value._fields, value))
            items.extend(flatten_dict(dict_items, key_, sep=sep).items())
        else:
            items.append((key_, value))
    return dict(items)


def copy_hparams(hparams: HParams):
    hp_vals = hparams.values()
    new_hparams = HParams(**hp_vals)
    # Make sure below params are part of new phparams while making a copy
    other_attrs = ["problem", "problem_hparams"]
    for attr in other_attrs:
        attr_val = getattr(hparams, attr, None)
        if attr_val is not None:
            setattr(new_hparams, attr, attr_val)
    return new_hparams


def create_hparams(hparams_set,
                   hparams_overrides_str="",
                   data_dir=None,
                   problem_name=None,
                   hparams_path=None):
    """Create HParams with data_dir and problem hparams, if kwargs provided."""
    hparams = registry.hparams(hparams_set)
    if hparams_path and tf.gfile.Exists(hparams_path):
        hparams = create_hparams_from_json(hparams_path, hparams)
    if data_dir:
        hparams.add_hparam("data_dir", data_dir)
    if hparams_overrides_str:
        tf.logging.info("Overriding hparams in %s with %s", hparams_set,
                        hparams_overrides_str)
        hparams = hparams.parse(hparams_overrides_str)
    if problem_name:
        add_problem_hparams(hparams, problem_name)
    return hparams


def create_hparams_from_json(json_path, hparams=None):
    """Loading hparams from json; can also start from hparams if specified."""
    tf.logging.info("Loading hparams from existing json %s" % json_path)
    with tf.gfile.Open(json_path, "r") as f:
        hparams_values = json.load(f)
        new_hparams = HParams(**hparams_values)
        # Some keys are in new_hparams but not hparams, so we need to be more
        #   careful than simply using parse_json() from HParams
        if hparams:  # hparams specified, so update values from json
            for key in sorted(new_hparams.values().keys()):
                if hasattr(hparams, key):  # Overlapped keys
                    value = getattr(hparams, key)
                    new_value = getattr(new_hparams, key)
                    if value != new_value:  # Different values
                        tf.logging.info("Overwrite key %s: %s -> %s" % (
                            key, value, new_value))
                        setattr(hparams, key, new_value)
        else:
            hparams = new_hparams

    return hparams


def add_problem_hparams(hparams, problem_name_or_instance):
    """Add problem hparams for the problems."""
    if isinstance(problem_name_or_instance, problem_lib.Problem):
        problem = problem_name_or_instance
    else:
        problem = registry.dataset(problem_name_or_instance)
    p_hparams = problem.get_hparams(hparams)
    hparams.problem = problem
    hparams.problem_hparams = p_hparams
