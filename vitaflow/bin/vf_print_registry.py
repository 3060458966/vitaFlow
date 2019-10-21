#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

"""
Run to see available datasets and models
python vf_print_registry.py
"""

import sys
import os
import warnings
from absl import logging
warnings.simplefilter(action='ignore', category=FutureWarning)
sys.path.append(os.getcwd())
logging.set_verbosity(logging.INFO)

from vitaflow.datasets import datasets # pylint: disable=unused-import
from vitaflow.models import models # pylint: disable=unused-import
from vitaflow.serving import serving # pylint: disable=unused-import

import vitaflow.utils.registry as registry


def maybe_log_registry_and_exit():
    logging.info(registry.help_string())
    sys.exit(0)


if __name__ == "__main__":
    # If we just have to print the registry, do that and exit early.
    maybe_log_registry_and_exit()
