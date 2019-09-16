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

"""Access vitaFlow Datasets."""

from vitaflow.datasets import all_datasets
from vitaflow.utils import registry
from vitaflow.datasets.interface_dataset import IDataset


def _dataset(name):
    return registry.dataset(name)


def available():
    return registry.list_base_datasets()


def get_dataset(name_or_instance):
    """Get dataset instance from problem name or instance"""
    if isinstance(name_or_instance, IDataset):
        ds = name_or_instance
    else:
        ds = _dataset(name_or_instance)
    return ds


all_datasets.import_modules(all_datasets.ALL_MODULES)
