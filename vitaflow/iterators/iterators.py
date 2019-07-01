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

"""Access vitaFlow data_iterators."""

from vitaflow.iterators import all_iterators
from vitaflow.utils import registry
from vitaflow.iterators.internal.iterator_base import DatasetInterface


def _data_iterator(name):
    return registry.data_iterator(name)


def available():
    return registry.list_base_data_iterators()


def get_data_iterator(name_or_instance):
    """Get data_iterator instance from problem name or instance"""
    if isinstance(name_or_instance, DatasetInterface):
        ds = name_or_instance
    else:
        ds = _data_iterator(name_or_instance)
    return ds


all_iterators.import_modules(all_iterators.ALL_MODULES)
