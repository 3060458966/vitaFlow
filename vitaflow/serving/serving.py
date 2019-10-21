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

"""Access vitaFlow serving classes."""

from vitaflow.serving import all_serving
from vitaflow.serving.iservice import IServing
from vitaflow.utils import registry


def _serving(name):
    return registry.serving(name)


def available():
    return registry.list_serving()


def get_serving(name_or_instance):
    """Get data_iterator instance from problem name or instance"""
    if isinstance(name_or_instance, IServing):
        serving_class = name_or_instance
    else:
        serving_class = _serving(name_or_instance)
    return serving_class


all_serving.import_modules(all_serving.ALL_MODULES)
