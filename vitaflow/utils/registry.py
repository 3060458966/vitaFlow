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

"""Object registration.

Registries are instances of `Registry`.

See `Registries` for a centralized list of object registries
(models, datasets, data iterators).

New functions and classes can be registered using `.register`. The can be
accessed/queried similar to dictionaries, keyed by default by `snake_case`
equivalents.

Forked from : https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/registry.py

```
@Registries.models.register
class MyModel(VfModel):
  ...

'my_model' in Registries.models  # True

for k in Registries.models:
  print(k)  # prints 'my_model'
_model = Registries.models['my_model'](constructor_arg)
```
"""

import collections

from vitaflow.utils import misc_utils
from tensorflow.python.util import tf_inspect as inspect  # pylint: disable=g-direct-tensorflow-import

def default_name(class_or_fn):
    """Default name for a class or function.

    This is the naming function by default for registries expecting classes or
    functions.

    Args:
      class_or_fn: class or function to be named.

    Returns:
      Default name for registration.
    """
    return misc_utils.camelcase_to_snakecase(class_or_fn.__name__)


default_object_name = lambda obj: default_name(type(obj))





def _nargs_validator(nargs, message):
    """Makes validator for function to ensure it takes nargs args."""
    if message is None:
        message = "Registered function must take exactly %d arguments" % nargs

    def f(key, value):
        del key
        spec = inspect.getfullargspec(value)
        if (len(spec.args) != nargs or spec.varargs is not None or
                spec.varkw is not None):
            raise ValueError(message)

    return f


DatasetSpec = collections.namedtuple("DatasetSpec",
                                     ["base_name", "was_reversed", "was_copy"])


def parse_dataset_name(name):
    """Determines if dataset_name specifies a copy and/or reversal.

    Args:
      name: str, dataset name, possibly with suffixes.

    Returns:
      datasetSpec: namedtuple with ["base_name", "was_reversed", "was_copy"]

    Raises:
      ValueError if name contains multiple suffixes of the same type
        ('_rev' or '_copy'). One of each is ok.
    """
    # Recursively strip tags until we reach a base name.
    if name.endswith("_rev"):
        base, was_reversed, was_copy = parse_dataset_name(name[:-4])
        if was_reversed:
            # duplicate rev
            raise ValueError(
                "Invalid dataset name %s: multiple '_rev' instances" % name)
        return DatasetSpec(base, True, was_copy)
    elif name.endswith("_copy"):
        base, was_reversed, was_copy = parse_dataset_name(name[:-5])
        if was_copy:
            raise ValueError(
                "Invalid dataset_name %s: multiple '_copy' instances" % name)
        return DatasetSpec(base, was_reversed, True)
    else:
        return DatasetSpec(name, False, False)


def get_dataset_name(base_name, was_reversed=False, was_copy=False):
    """Construct a dataset name from base and reversed/copy options.

    Inverse of `parse_dataset_name`.

    Args:
      base_name: base dataset name. Should not end in "_rev" or "_copy"
      was_reversed: if the dataset is to be reversed
      was_copy: if the dataset is to be copied

    Returns:
      string name consistent with use with `parse_dataset_name`.

    Raises:
      ValueError if `base_name` ends with "_rev" or "_copy"
    """
    if any(base_name.endswith(suffix) for suffix in ("_rev", "_copy")):
        raise ValueError("`base_name` cannot end in '_rev' or '_copy'")
    name = base_name
    if was_copy:
        name = "%s_copy" % name
    if was_reversed:
        name = "%s_rev" % name
    return name


def _dataset_name_validator(k, v):
    del v
    if parse_dataset_name(k).base_name != k:
        raise KeyError(
            "Invalid dataset name: cannot end in %s or %s" % ("_rev", "_copy"))


def _on_model_set(k, v):
    v.REGISTERED_NAME = k


def _on_dataset_set(k, v):
    v.name = k


def _on_serving_set(k, v):
    v.name = k


def _call_value(k, v):
    del k
    return v()


class Registry(object):
    """Dict-like class for managing function registrations.

    ```python
    my_registry = Registry("custom_name")

    @my_registry.register
    def my_func():
      pass

    @my_registry.register()
    def another_func():
      pass

    @my_registry.register("non_default_name")
    def third_func(x, y, z):
      pass

    def foo():
      pass

    my_registry.register()(foo)
    my_registry.register("baz")(lambda (x, y): x + y)
    my_register.register("bar")

    print(list(my_registry))
    # ["my_func", "another_func", "non_default_name", "foo", "baz"]
    # (order may vary)
    print(my_registry["non_default_name"] is third_func)  # True
    print("third_func" in my_registry)                    # False
    print("bar" in my_registry)                           # False
    my_registry["non-existent_key"]                       # raises KeyError
    ```

    Optional validation, on_set callback and value transform also supported.
    See `__init__` doc.
    """

    def __init__(self,
                 registry_name,
                 default_key_fn=default_name,
                 validator=None,
                 on_set=None,
                 value_transformer=(lambda k, v: v)):
        """Construct a new registry.

        Args:
          registry_name: str identifier for the given registry. Used in error msgs.
          default_key_fn (optional): function mapping value -> key for registration
            when a key is not provided
          validator (optional): if given, this is run before setting a given (key,
            value) pair. Accepts (key, value) and should raise if there is a
            dataset. Overwriting existing keys is not allowed and is checked
            separately. Values are also checked to be callable separately.
          on_set (optional): callback function accepting (key, value) pair which is
            run after an item is successfully set.
          value_transformer (optional): if run, `__getitem__` will return
            value_transformer(key, registered_value).
        """
        self._registry = {}
        self._name = registry_name
        self._default_key_fn = default_key_fn
        self._validator = validator
        self._on_set = on_set
        self._value_transformer = value_transformer

    def default_key(self, value):
        """Default key used when key not provided. Uses function from __init__."""
        return self._default_key_fn(value)

    @property
    def name(self):
        return self._name

    def validate(self, key, value):
        """Validation function run before setting. Uses function from __init__."""
        if self._validator is not None:
            self._validator(key, value)

    def on_set(self, key, value):
        """Callback called on successful set. Uses function from __init__."""
        if self._on_set is not None:
            self._on_set(key, value)

    def __setitem__(self, key, value):
        """Validate, set, and (if successful) call `on_set` for the given item.

        Args:
          key: key to store value under. If `None`, `self.default_key(value)` is
            used.
          value: callable stored under the given key.

        Raises:
          KeyError: if key is already in registry.
        """
        if key is None:
            key = self.default_key(value)
        if key in self:
            raise KeyError(
                "key %s already registered in registry %s" % (key, self._name))
        if not callable(value):
            raise ValueError("value must be callable")
        self.validate(key, value)
        self._registry[key] = value
        self.on_set(key, value)

    def register(self, key_or_value=None):
        """Decorator to register a function, or registration itself.

        This is primarily intended for use as a decorator, either with or without
        a key/parentheses.
        ```python
        @my_registry.register('key1')
        def value_fn(x, y, z):
          pass

        @my_registry.register()
        def another_fn(x, y):
          pass

        @my_registry.register
        def third_func():
          pass
        ```

        Note if key_or_value is provided as a non-callable, registration only
        occurs once the returned callback is called with a callable as its only
        argument.
        ```python
        callback = my_registry.register('different_key')
        'different_key' in my_registry  # False
        callback(lambda (x, y): x + y)
        'different_key' in my_registry  # True
        ```

        Args:
          key_or_value (optional): key to access the registered value with, or the
            function itself. If `None` (default), `self.default_key` will be called
            on `value` once the returned callback is called with `value` as the only
            arg. If `key_or_value` is itself callable, it is assumed to be the value
            and the key is given by `self.default_key(key)`.

        Returns:
          decorated callback, or callback generated a decorated function.
        """
        def decorator(value, key):
            self[key] = value
            return value

        # Handle if decorator was used without parens
        if callable(key_or_value):
            return decorator(value=key_or_value, key=None)
        else:
            return lambda value: decorator(value, key=key_or_value)

    def __getitem__(self, key):
        if key not in self:
            raise KeyError("%s never registered with registry %s. Available:\n %s" %
                           (key, self.name, display_list_by_prefix(sorted(self), 4)))
        value = self._registry[key]
        return self._value_transformer(key, value)

    def __contains__(self, key):
        return key in self._registry

    def keys(self):
        return self._registry.keys()

    def values(self):
        return (self[k] for k in self)  # complicated because of transformer

    def items(self):
        return ((k, self[k]) for k in self)  # complicated because of transformer

    def __iter__(self):
        return iter(self._registry)

    def __len__(self):
        return len(self._registry)

    def _clear(self):
        self._registry.clear()

    def get(self, key, default=None):
        return self[key] if key in self else default


class Registries(object):
    """Object holding `Registry` objects."""

    def __init__(self):
        raise RuntimeError("Registries is not intended to be instantiated")

    models = Registry("models", on_set=_on_model_set)

    datasets = Registry("datasets", validator=_dataset_name_validator, on_set=_on_dataset_set)

    serving = Registry("serving", on_set=_on_serving_set)


register_model = Registries.models.register
list_models = lambda: sorted(Registries.models)

register_dataset = Registries.datasets.register
list_datasets = lambda: sorted(Registries.datasets)

register_serving = Registries.serving.register
list_serving = lambda: sorted(Registries.serving)

def dataset(dataset_name):
    """Get  dataset registered in `base_registry`.

    Args:
      dataset_name: string dataset name.

    Returns:
      dataset registered in the given  registry.
    """
    return Registries.datasets[dataset_name]


def model(model_name):
    """Get  dataset registered in `base_registry`.

    Args:
      dataset_name: string dataset name.

    Returns:
      dataset registered in the given  registry.
    """
    return Registries.models[model_name]



def serving(serving_class_name):
    """Get  dataset registered in `base_registry`.

    Args:
      dataset_name: string dataset name.

    Returns:
      dataset registered in the given  registry.
    """
    return Registries.serving[serving_class_name]

def display_list_by_prefix(names_list, starting_spaces=0):
    """Creates a help string for names_list grouped by prefix."""
    cur_prefix, result_lines = None, []
    space = " " * starting_spaces
    for name in sorted(names_list):
        split = name.split("_", 1)
        prefix = split[0]
        if cur_prefix != prefix:
            result_lines.append(space + prefix + ":")
            cur_prefix = prefix
        result_lines.append(space + "  * " + name)
    return "\n".join(result_lines)


def help_string():
    """Generate help string with contents of registry."""
    help_str = """
Registry contents:
------------------

Models:
%s

Datasets:
%s

Serving:
%s  
"""
    lists = tuple(
        display_list_by_prefix(entries, starting_spaces=4) for entries in [  # pylint: disable=g-complex-comprehension
            list_models(),
            list_datasets(),
            list_serving()
        ])
    return help_str % lists
