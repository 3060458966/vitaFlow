# coding=utf-8
# Copyright 2019 The vitaFlow Authors. All Rights Reserved.
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
# Forked from https://github.com/tensorflow/tensor2tensor
"""
Hyperparameter manager
Combination of
- https://github.com/asyml/texar/blob/master/texar/hyperparams.py
- https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/hparam.py
"""

import os
import copy
import json
import numbers
import re
import six


# Define the regular expression for parsing a single clause of the input
# (delimited by commas).  A legal clause looks like:
#   <variable name>[<index>]? = <rhs>
# where <rhs> is either a single token or [] enclosed list of tokens.
# For example:  "var[1] = a" or "x = [1,2,3]"
from vitaflow.utils.dtypes import is_callable
from vitaflow.utils.print_helper import print_info, print_error


def _type_name(value):
    return type(value).__name__

PARAM_RE = re.compile(r"""
  (?P<name>[a-zA-Z][\w\.]*)      # variable name: "var" or "x"
  (\[\s*(?P<index>\d+)\s*\])?  # (optional) index: "1" or None
  \s*=\s*
  ((?P<val>[^,\[]*)            # single value: "a" or None
   |
   \[(?P<vals>[^\]]*)\])       # list of values: None or "1,2,3"
  ($|,\s*)""", re.VERBOSE)


def _parse_fail(name, var_type, value, values):
    """Helper function for raising a value error for bad assignment."""
    raise ValueError(
        'Could not parse hparam \'%s\' of type \'%s\' with value \'%s\' in %s' %
        (name, var_type.__name__, value, values))


def _reuse_fail(name, values):
    """Helper function for raising a value error for reuse of name."""
    raise ValueError('Multiple assignments to variable \'%s\' in %s' % (name,
                                                                        values))


def _process_scalar_value(name, parse_fn, var_type, m_dict, values,
                          results_dictionary):
    """Update results_dictionary with a scalar value.

    Used to update the results_dictionary to be returned by parse_values when
    encountering a clause with a scalar RHS (e.g.  "s=5" or "arr[0]=5".)

    Mutates results_dictionary.

    Args:
      name: Name of variable in assignment ("s" or "arr").
      parse_fn: Function for parsing the actual value.
      var_type: Type of named variable.
      m_dict: Dictionary constructed from regex parsing.
        m_dict['val']: RHS value (scalar)
        m_dict['index']: List index value (or None)
      values: Full expression being parsed
      results_dictionary: The dictionary being updated for return by the parsing
        function.

    Raises:
      ValueError: If the name has already been used.
    """
    try:
        parsed_value = parse_fn(m_dict['val'])
    except ValueError:
        _parse_fail(name, var_type, m_dict['val'], values)

    # If no index is provided
    if not m_dict['index']:
        if name in results_dictionary:
            _reuse_fail(name, values)
        results_dictionary[name] = parsed_value
    else:
        if name in results_dictionary:
            # The name has already been used as a scalar, then it
            # will be in this dictionary and map to a non-dictionary.
            if not isinstance(results_dictionary.get(name), dict):
                _reuse_fail(name, values)
        else:
            results_dictionary[name] = {}

        index = int(m_dict['index'])
        # Make sure the index position hasn't already been assigned a value.
        if index in results_dictionary[name]:
            _reuse_fail('{}[{}]'.format(name, index), values)
        results_dictionary[name][index] = parsed_value


def _process_list_value(name, parse_fn, var_type, m_dict, values,
                        results_dictionary):
    """Update results_dictionary from a list of values.

    Used to update results_dictionary to be returned by parse_values when
    encountering a clause with a list RHS (e.g.  "arr=[1,2,3]".)

    Mutates results_dictionary.

    Args:
      name: Name of variable in assignment ("arr").
      parse_fn: Function for parsing individual values.
      var_type: Type of named variable.
      m_dict: Dictionary constructed from regex parsing.
        m_dict['val']: RHS value (scalar)
      values: Full expression being parsed
      results_dictionary: The dictionary being updated for return by the parsing
        function.

    Raises:
      ValueError: If the name has an index or the values cannot be parsed.
    """
    if m_dict['index'] is not None:
        raise ValueError('Assignment of a list to a list index.')
    elements = filter(None, re.split('[ ,]', m_dict['vals']))
    # Make sure the name hasn't already been assigned a value
    if name in results_dictionary:
        raise _reuse_fail(name, values)
    try:
        results_dictionary[name] = [parse_fn(e) for e in elements]
    except ValueError:
        _parse_fail(name, var_type, m_dict['vals'], values)


def _cast_to_type_if_compatible(name, param_type, value):
    """Cast hparam to the provided type, if compatible.

    Args:
      name: Name of the hparam to be cast.
      param_type: The type of the hparam.
      value: The value to be cast, if compatible.

    Returns:
      The result of casting `value` to `param_type`.

    Raises:
      ValueError: If the type of `value` is not compatible with param_type.
        * If `param_type` is a string type, but `value` is not.
        * If `param_type` is a boolean, but `value` is not, or vice versa.
        * If `param_type` is an integer type, but `value` is not.
        * If `param_type` is a float type, but `value` is not a numeric type.
    """
    fail_msg = (
            "Could not cast hparam '%s' of type '%s' from value %r" %
            (name, param_type, value))

    # Some callers use None, for which we can't do any casting/checking. :(
    if issubclass(param_type, type(None)):
        return value

    # Avoid converting a non-string type to a string.
    if (issubclass(param_type, (six.string_types, six.binary_type)) and
            not isinstance(value, (six.string_types, six.binary_type))):
        raise ValueError(fail_msg)

    # Avoid converting a number or string type to a boolean or vice versa.
    if issubclass(param_type, bool) != isinstance(value, bool):
        raise ValueError(fail_msg)

    # Avoid converting float to an integer (the reverse is fine).
    if (issubclass(param_type, numbers.Integral) and
            not isinstance(value, numbers.Integral)):
        raise ValueError(fail_msg)

    # Avoid converting a non-numeric type to a numeric type.
    if (issubclass(param_type, numbers.Number) and
            not isinstance(value, numbers.Number)):
        raise ValueError(fail_msg)

    return param_type(value)


def parse_values(values, type_map, ignore_unknown=False):
    """Parses hyperparameter values from a string into a python map.

    `values` is a string containing comma-separated `name=value` pairs.
    For each pair, the value of the hyperparameter named `name` is set to
    `value`.

    If a hyperparameter name appears multiple times in `values`, a ValueError
    is raised (e.g. 'a=1,a=2', 'a[1]=1,a[1]=2').

    If a hyperparameter name in both an index assignment and scalar assignment,
    a ValueError is raised.  (e.g. 'a=[1,2,3],a[0] = 1').

    The hyperparameter name may contain '.' symbols, which will result in an
    attribute name that is only accessible through the getattr and setattr
    functions.  (And must be first explicit added through add_hparam.)

    WARNING: Use of '.' in your variable names is allowed, but is not well
    supported and not recommended.

    The `value` in `name=value` must follows the syntax according to the
    type of the parameter:

    *  Scalar integer: A Python-parsable integer point value.  E.g.: 1,
       100, -12.
    *  Scalar float: A Python-parsable floating point value.  E.g.: 1.0,
       -.54e89.
    *  Boolean: Either true or false.
    *  Scalar string: A non-empty sequence of characters, excluding comma,
       spaces, and square brackets.  E.g.: foo, bar_1.
    *  List: A comma separated list of scalar values of the parameter type
       enclosed in square brackets.  E.g.: [1,2,3], [1.0,1e-12], [high,low].

    When index assignment is used, the corresponding type_map key should be the
    list name.  E.g. for "arr[1]=0" the type_map must have the key "arr" (not
    "arr[1]").

    Args:
      values: String.  Comma separated list of `name=value` pairs where
        'value' must follow the syntax described above.
      type_map: A dictionary mapping hyperparameter names to types.  Note every
        parameter name in values must be a key in type_map.  The values must
        conform to the types indicated, where a value V is said to conform to a
        type T if either V has type T, or V is a list of elements of type T.
        Hence, for a multidimensional parameter 'x' taking float values,
        'x=[0.1,0.2]' will parse successfully if type_map['x'] = float.
      ignore_unknown: Bool. Whether values that are missing a type in type_map
        should be ignored. If set to True, a ValueError will not be raised for
        unknown hyperparameter type.

    Returns:
      A python map mapping each name to either:
      * A scalar value.
      * A list of scalar values.
      * A dictionary mapping index numbers to scalar values.
      (e.g. "x=5,L=[1,2],arr[1]=3" results in {'x':5,'L':[1,2],'arr':{1:3}}")

    Raises:
      ValueError: If there is a problem with input.
      * If `values` cannot be parsed.
      * If a list is assigned to a list index (e.g. 'a[1] = [1,2,3]').
      * If the same rvalue is assigned two different values (e.g. 'a=1,a=2',
        'a[1]=1,a[1]=2', or 'a=1,a=[1]')
    """
    results_dictionary = {}
    pos = 0
    while pos < len(values):
        m = PARAM_RE.match(values, pos)
        if not m:
            raise ValueError('Malformed hyperparameter value: %s' % values[pos:])
        # Check that there is a comma between parameters and move past it.
        pos = m.end()
        # Parse the values.
        m_dict = m.groupdict()
        name = m_dict['name']
        if name not in type_map:
            if ignore_unknown:
                continue
            raise ValueError('Unknown hyperparameter type for %s' % name)
        type_ = type_map[name]

        # Set up correct parsing function (depending on whether type_ is a bool)
        if type_ == bool:

            def parse_bool(value):
                if value in ['true', 'True']:
                    return True
                elif value in ['false', 'False']:
                    return False
                else:
                    try:
                        return bool(int(value))
                    except ValueError:
                        _parse_fail(name, type_, value, values)

            parse = parse_bool
        else:
            parse = type_

        # If a singe value is provided
        if m_dict['val'] is not None:
            _process_scalar_value(name, parse, type_, m_dict, values,
                                  results_dictionary)

        # If the assigned value is a list:
        elif m_dict['vals'] is not None:
            _process_list_value(name, parse, type_, m_dict, values,
                                results_dictionary)

        else:  # Not assigned a list or value
            _parse_fail(name, type_, '', values)

    return results_dictionary


class HParams(object):
    """A class that maintains hyperparameters for configing vitaFlow modules.
    The class has several useful features:

    - **Auto-completion of missing values.** Users can specify only a subset of\
    hyperparameters they care about. Other hyperparameters will automatically\
    take the default values. The auto-completion performs **recursively** so \
    that hyperparameters taking `dict` values will also be auto-completed \
    **All vitaFlow modules** provide a \
    :meth:`default_hparams` containing allowed hyperparameters and their \
    default values. For example

        .. code-block:: python

            ## Recursive auto-completion
            default_hparams = {"a": 1, "b": {"c": 2, "d": 3}}
            hparams = {"b": {"c": 22}}
            hparams_ = HParams(hparams, default_hparams)
            hparams_.todict() == {"a": 1, "b": {"c": 22, "d": 3}}
                # "a" and "d" are auto-completed

            ## All vitaFlow modules have built-in `default_hparams`
            hparams = {"dropout_rate": 0.1}
            conv = vf.models.image.classification.NaiveConvNet(hparams=hparams, ...)
            conv.hparams.todict() == {
                "dropout_rate": 0.1,  # provided value
                "dim": 100            # default value
                
            }

    - **Automatic typecheck.** For most hyperparameters, provided value must\
    have the same or compatible dtype with the default value. HParams does\
    necessary typecheck, and raises Error if improper dtype is provided.\
    Also, hyperparameters not listed in `default_hparams` are not allowed,\
    except for "kwargs" as detailed below.

    - **Flexible dtype for specified hyperparameters.**  Some hyperparameters\
    may allow different dtypes of values.

        - Hyperparameters named "type" are not typechecked.\
        For example, in :func:`~texar.core.get_rnn_cell`, hyperparameter \
        `"type"` can take value of an RNNCell class, its string name of module \
        path, or an RNNCell class instance. (String name or module path is \
        allowd so that users can specify the value in YAML config files.)

        - For other hyperparameters, list them\
        in the "@no_typecheck" field in `default_hparams` to skip typecheck. \
        For example, in :func:`~texar.core.get_rnn_cell`, hyperparameter \
        "_keep_prob" can be set to either a `float` or a `tf.placeholder`.

    - **Special flexibility of keyword argument hyparameters.** \
    Hyperparameters named "kwargs" are used as keyword arguments for a class\
    constructor or a function call. Such hyperparameters take a `dict`, and \
    users can add arbitrary valid keyword arguments to the dict. For example:

        .. code-block:: python

            default_rnn_cell_hparams = {
                "type": "BasicLSTMCell",
                "kwargs": { "num_units": 256 }
                # Other hyperparameters
                
            }
            my_hparams = {
                "kwargs" {
                    "num_units": 123,
                    "forget_bias": 0.0         # Other valid keyword arguments
                    "activation": "tf.nn.relu" # for BasicLSTMCell constructor
                }
            }
            _ = HParams(my_hparams, default_rnn_cell_hparams)

    - **Rich interfaces.** An HParams instance provides rich interfaces for\
    accessing, updating, or adding hyperparameters.

        .. code-block:: python

            hparams = HParams(my_hparams, default_hparams)
            # Access
            hparams.type == hparams["type"]
            # Update
            hparams.type = "GRUCell"
            hparams.kwargs = { "num_units": 100 }
            hparams.kwargs.num_units == 100
            # Add new
            hparams.add_hparam("index", 1)
            hparams.index == 1

            # Convert to `dict` (recursively)
            type(hparams.todic()) == dict

            # I/O
            pickle.dump(hparams, "hparams.dump")
            with open("hparams.dump", 'rb') as f:
                hparams_loaded = pickle.load(f)


    Args:
        hparams: A `dict` or an `HParams` instance containing hyperparameters.
            If `None`, all hyperparameters are set to default values.
        default_hparams (dict): Hyperparameters with default values. If `None`,
            Hyperparameters are fully defined by :attr:`hparams`.
        allow_new_hparam (bool): If `False` (default), :attr:`hparams` cannot
            contain hyperparameters that are not included in
            :attr:`default_hparams`, except for the case of :attr:`"kwargs"` as
            above.
    """

    # - The default hyperparameters in :attr:`"kwargs"` are used (for typecheck\
    # and complementing missing hyperparameters) only when :attr:`"type"` \
    # takes default value (i.e., missing in :attr:`hparams` or set to \
    # the same value with the default). In this case :attr:`kwargs` allows to \
    # contain new keys not included in :attr:`default_hparams["kwargs"]`.
    #
    # - If :attr:`"type"` is set to an other \
    # value and :attr:`"kwargs"` is missing in :attr:`hparams`, \
    # :attr:`"kwargs"` is set to an empty dictionary.

    def __init__(self, hparams=None, default_hparams=None, allow_new_hparam=False, **kwargs):

        _hparam_types = {}
        parsed_hparams = {}

        if hparams is None and default_hparams is None:
            for name, value in six.iteritems(kwargs):
                parsed_hparams[name] = value
            _hparam_types = self.add_hparam_type(parsed_hparams)
        else:
            if isinstance(hparams, HParams):
                hparams = hparams.todict()
            if default_hparams is not None:
                parsed_hparams = self._parse(hparams, default_hparams, allow_new_hparam)
            else:
                parsed_hparams = self._parse(hparams, hparams)
            _hparam_types = self.add_hparam_type(parsed_hparams)

        super(HParams, self).__setattr__('_hparams', parsed_hparams)
        super(HParams, self).__setattr__('_hparam_types', _hparam_types)


    def __getattr__(self, name):
        """Retrieves the value of the hyperparameter.
        """
        if name == '_hparams':
            return super(HParams, self).__getattribute__('_hparams')
        if name == '_hparam_types':
            return super(HParams, self).__getattribute__('_hparam_types')
        if name not in self._hparams:
            # Raise AttributeError to allow copy.deepcopy, etc
            raise AttributeError("Unknown hyperparameter: %s" % name)

        value = self._hparams[name]
        return value

    def __getitem__(self, name):
        """Retrieves the value of the hyperparameter.
        """
        return self.__getattr__(name)

    def __setitem__(self, key, value):
        self.__setattr__(key, value)

    def __setattr__(self, name, value):
        """Set the value of an existing hyperparameter.

        This function verifies that the type of the value matches the type of the
        existing hyperparameter.

        Args:
          name: Name of the hyperparameter.
          value: New value of the hyperparameter.

        Raises:
          KeyError: If the hyperparameter doesn't exist.
          ValueError: If there is a type mismatch.
        """
        if name not in self._hparams:
            print(
                "Unknown hyperparameter: %s. Only the `kwargs` "
                "hyperparameters can contain new entries undefined "
                "in default hyperparameters." % name) #TODO in texar this is an Error
        parsed_value = self._parse_value(value, name)
        self._hparams[name] = parsed_value

    def __iter__(self):
        for name, value in self._hparams.items():
            yield name, value

    def __len__(self):
        return len(self._hparams)

    def __contains__(self, name):
        return name in self._hparams

    def __str__(self):
        return str(self.values())
        #return str(sorted(self.values().items()))

    def keys(self):
        """Returns the list of hyperparam names
        """
        return self._hparams.keys()

    def values(self):
        """Return the hyperparameter values as a Python dictionary.

        Returns:
          A dictionary with hyperparameter names as keys.  The values are the
          hyperparameter values.
        """
        return self.todict()

    def items(self): #TODO add test case
        """Returns the list of hyperparam `(name, value)` pairs
        """
        return list(self._hparams.items())#iter(self)

    def todict(self):
        """Returns a copy of hyperparameters as a dictionary.
        """
        dict_ = copy.deepcopy(self._hparams)
        for name, value in self._hparams.items():
            if isinstance(value, HParams):
                dict_[name] = value.todict()
        return dict_

    def add_hparam_type(self, parsed_hparams):
        """Adds {name, value} pair value type.

        Args:
          name: Name of the hyperparameter.
          value: Value of the hyperparameter. Can be one of the following types:
            int, float, string, int list, float list, or string list.

        Raises:
          ValueError: if one of the arguments is invalid.
        """
        # Keys in kwargs are unique, but 'name' could the name of a pre-existing
        # attribute of this object.  In that case we refuse to use it as a
        # hyperparameter name.
        _hparam_types = {}
        for name, value in six.iteritems(parsed_hparams):
            if getattr(self, name, None) is not None:
                raise ValueError('Hyperparameter name is reserved: %s' % name)
            if isinstance(value, (list, tuple)):
                if not value:
                    raise ValueError(
                        'Multi-valued hyperparameters cannot be empty: %s' % name)
                _hparam_types[name] = (type(value[0]), True)
            else:
                _hparam_types[name] = (type(value), False)
        return _hparam_types


    def _parse_value(self, value, name=None):
        """
        Note: Since we encode all values as HParams,
        there is a need for explicit dict conversion to use with `six`
        eg: six.itervalues(self.hparams.some_dict_option_key.todict())
        Args:
            value:
            name:

        Returns:

        """
        if isinstance(value, dict) and (name is None or name != "kwargs"):
            value = HParams(value, None)
            return value
        else:
            return value

    def parse(self, values):
        """Override existing hyperparameter values, parsing new values from a string.

        See parse_values for more detail on the allowed format for values.

        Args:
          values: String.  Comma separated list of `name=value` pairs where 'value'
            must follow the syntax described above.

        Returns:
          The `HParams` instance.

        Raises:
          ValueError: If `values` cannot be parsed or a hyperparameter in `values`
          doesn't exist.
        """
        type_map = dict()
        for name, t in self._hparam_types.items():
            param_type, _ = t
            type_map[name] = param_type
        values_map = parse_values(values, type_map)
        return self.override_from_dict(values_map)

    def override_from_dict(self, values_dict):
        """Override existing hyperparameter values, parsing new values from a dictionary.

        Args:
          values_dict: Dictionary of name:value pairs.

        Returns:
          The `HParams` instance.

        Raises:
          KeyError: If a hyperparameter in `values_dict` doesn't exist.
          ValueError: If `values_dict` cannot be parsed.
        """
        for name, value in values_dict.items():
            self.set_hparam(name, value)
        return self

    def get(self, name, default=None):
        """Returns the hyperparameter value for the given name. If name is not
        available then returns :attr:`default`.

        Args:
            name (str): the name of hyperparameter.
            default: the value to be returned in case name does not exist.
        """
        if name in self._hparam_types:
            # Ensure that default is compatible with the parameter type.
            if default is not None:
                param_type, is_param_list = self._hparam_types[name]
                type_str = 'list<%s>' % param_type if is_param_list else str(param_type)
                fail_msg = ("Hparam '%s' of type '%s' is incompatible with "
                            'default=%s' % (name, type_str, default))

                is_default_list = isinstance(default, list)
                if is_param_list != is_default_list:
                    raise ValueError(fail_msg)

                try:
                    if is_default_list:
                        for value in default:
                            _cast_to_type_if_compatible(name, param_type, value)
                    else:
                        _cast_to_type_if_compatible(name, param_type, default)
                except ValueError as e:
                    raise ValueError('%s. %s' % (fail_msg, e))

            return self.__getattr__(name)

        return default

    def add_hparam(self, name, value):
        """Adds a new hyperparameter.
        """
        # if (name in self._hparams) or hasattr(self, name):
        #     raise ValueError("Hyperparameter name already exists: %s" % name)
        if getattr(self, name, None) is not None:
            raise ValueError('Hyperparameter name is reserved: %s' % name)
        self._hparams[name] = self._parse_value(value, name)

        if isinstance(value, (list, tuple)):
            if not value:
                raise ValueError(
                    'Multi-valued hyperparameters cannot be empty: %s' % name)
            self._hparam_types[name] = (type(value[0]), True)
        else:
            self._hparam_types[name] = (type(value), False)



    def set_hparam(self, name, value):
        """Set the value of an existing hyperparameter.

        This function verifies that the type of the value matches the type of the
        existing hyperparameter.

        Args:
          name: Name of the hyperparameter.
          value: New value of the hyperparameter.

        Raises:
          KeyError: If the hyperparameter doesn't exist.
          ValueError: If there is a type mismatch.
        """
        param_type, is_list = self._hparam_types[name]
        if isinstance(value, list):
            if not is_list:
                raise ValueError(
                    'Must not pass a list for single-valued parameter: %s' % name)
            # setattr(self, name, [
            #     _cast_to_type_if_compatible(name, param_type, v) for v in value])
            self._hparams[name] = [_cast_to_type_if_compatible(name, param_type, v) for v in value]
        else:
            if is_list:
                raise ValueError(
                    'Must pass a list for multi-valued parameter: %s.' % name)
            # setattr(self, name, _cast_to_type_if_compatible(name, param_type, value))
            self._hparams[name] = _cast_to_type_if_compatible(name, param_type, value)

    def del_hparam(self, name):
        """Removes the hyperparameter with key 'name'.

        Does nothing if it isn't present.

        Args:
          name: Name of the hyperparameter.
        """
        if self._hparams[name]:
            del self._hparam_types[name]
            del self._hparams[name]

    def as_json(self):
        """Return a string of the hparams.
        """
        hparams_dict = self.todict()
        return json.dumps(hparams_dict, sort_keys=True, indent=2)

    def to_json(self, indent=None, separators=None, sort_keys=False):
        """Serializes the hyperparameters into JSON.

        Args:
          indent: If a non-negative integer, JSON array elements and object members
            will be pretty-printed with that indent level. An indent level of 0, or
            negative, will only insert newlines. `None` (the default) selects the
            most compact representation.
          separators: Optional `(item_separator, key_separator)` tuple. Default is
            `(', ', ': ')`.
          sort_keys: If `True`, the output dictionaries will be sorted by key.

        Returns:
          A JSON string.
        """
        def remove_callables(x):
            """Omit callable elements from input with arbitrary nesting."""
            if isinstance(x, dict):
                return {k: remove_callables(v) for k, v in six.iteritems(x)
                        if not callable(v)}
            elif isinstance(x, list):
                return [remove_callables(i) for i in x if not callable(i)]
            else:
                try:
                    json.dumps(x)
                    return x #TODO for some resons we are having class objects getting leaked into the hparams
                except:
                    return str(x)

        return json.dumps(
            #remove_callables(self.values()),
            {k: v.__name__ if callable(v) else v for k, v in six.iteritems(self.values())},
            indent=indent,
            separators=separators,
            sort_keys=sort_keys)

    def parse_json(self, values_json):
        """Override existing hyperparameter values, parsing new values from a json object.

        Args:
          values_json: String containing a json object of name:value pairs.

        Returns:
          The `HParams` instance.

        Raises:
          KeyError: If a hyperparameter in `values_json` doesn't exist.
          ValueError: If `values_json` cannot be parsed.
        """
        values_map = json.loads(values_json)
        return self.override_from_dict(values_map)


    # @staticmethod
    def _parse(self,
               hparams,  # pylint: disable=too-many-branches, too-many-statements
               default_hparams,
               allow_new_hparam=False):
        """Parses hyperparameters.

        Args:
            hparams (dict): Hyperparameters. If `None`, all hyperparameters are
                set to default values.
            default_hparams (dict): Hyperparameters with default values.
                If `None`,Hyperparameters are fully defined by :attr:`hparams`.
            allow_new_hparam (bool): If `False` (default), :attr:`hparams`
                cannot contain hyperparameters that are not included in
                :attr:`default_hparams`, except the case of :attr:`"kwargs"`.

        Return:
            A dictionary of parsed hyperparameters. Returns `None` if both
            :attr:`hparams` and :attr:`default_hparams` are `None`.

        Raises:
            ValueError: If :attr:`hparams` is not `None` and
                :attr:`default_hparams` is `None`.
            ValueError: If :attr:`default_hparams` contains "kwargs" not does
                not contains "type".
        """
        if hparams is None and default_hparams is None:
            return None

        if hparams is None:
            return self._parse(default_hparams, default_hparams)

        if default_hparams is None:
            raise ValueError("`default_hparams` cannot be `None` if `hparams` "
                             "is not `None`.")
        no_typecheck_names = default_hparams.get("@no_typecheck", [])

        if "kwargs" in default_hparams and "type" not in default_hparams:
            raise ValueError("Ill-defined hyperparameter structure: 'kwargs' "
                             "must accompany with 'type'.")

        parsed_hparams = copy.deepcopy(default_hparams)

        # Parse recursively for params of type dictionary that are missing
        # in `hparams`.
        for name, value in default_hparams.items():
            if name not in hparams and isinstance(value, dict):
                if name == "kwargs" and "type" in hparams and \
                        hparams["type"] != default_hparams["type"]:
                    # Set params named "kwargs" to empty dictionary if "type"
                    # takes value other than default.
                    parsed_hparams[name] = HParams(hparams={}, default_hparams={})
                else:
                    parsed_hparams[name] = HParams(hparams=value, default_hparams=value)

        # Parse hparams
        for name, value in hparams.items():
            if name not in default_hparams:
                if allow_new_hparam:
                    parsed_hparams[name] = self._parse_value(value, name)
                    continue
                else:
                    raise ValueError(
                        "Unknown hyperparameter: %s. Only hyperparameters "
                        "named 'kwargs' hyperparameters can contain new "
                        "entries undefined in default hyperparameters. Check your hparams!" % name)

            if value is None:
                parsed_hparams[name] = \
                    self._parse_value(parsed_hparams[name])

            default_value = default_hparams[name]
            if default_value is None:
                parsed_hparams[name] = self._parse_value(value)
                continue

            # Parse recursively for params of type dictionary.
            if isinstance(value, dict):
                if name not in no_typecheck_names \
                        and not isinstance(default_value, dict):
                    raise ValueError(
                        "Hyperparameter '%s' must have type %s, got %s" %
                        (name, _type_name(default_value), _type_name(value)))
                if name == "kwargs":
                    if "type" in hparams and \
                            hparams["type"] != default_hparams["type"]:
                        # Leave "kwargs" as-is if "type" takes value
                        # other than default.
                        parsed_hparams[name] = HParams(hparams=value, default_hparams=value)
                    else:
                        # Allow new hyperparameters if "type" takes default
                        # value
                        parsed_hparams[name] = HParams(
                            hparams=value, default_hparams=default_value, allow_new_hparam=True)
                elif name in no_typecheck_names:
                    parsed_hparams[name] = HParams(hparams=value, default_hparams=value)
                else:
                    parsed_hparams[name] = HParams(
                        hparams=value, default_hparams=default_value, allow_new_hparam=allow_new_hparam)
                continue

            # Do not type-check hyperparameter named "type" and accompanied
            # with "kwargs"
            if name == "type" and "kwargs" in default_hparams:
                parsed_hparams[name] = value
                continue

            if name in no_typecheck_names:
                parsed_hparams[name] = value
            elif isinstance(value, type(default_value)):
                parsed_hparams[name] = value
            elif is_callable(value) and is_callable(default_value):
                parsed_hparams[name] = value
            else:
                try:
                    parsed_hparams[name] = type(default_value)(value)
                except TypeError:
                    raise ValueError(
                        "Hyperparameter '%s' must have type %s, got %s" %
                        (name, _type_name(default_value), _type_name(value)))

        return parsed_hparams