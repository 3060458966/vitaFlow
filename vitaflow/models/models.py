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

from vitaflow.models import all_models
from vitaflow.utils import registry
from vitaflow.models.internals.model_base import ModelBase


def _model(name):
    return registry.model(name)


def available():
    return registry.list_models()


def get_model(name_or_instance):
    """Get data_iterator instance from problem name or instance"""
    if isinstance(name_or_instance, ModelBase):
        ds = name_or_instance
    else:
        ds = _model(name_or_instance)
    return ds


all_models.import_modules(all_models.ALL_MODULES)
