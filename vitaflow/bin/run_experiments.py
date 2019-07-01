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
Run Tensorflow Experiments for given dataset, iterator and model

run_experiments \
	--mode=train \
	--config=path/to/gin_config.gin
"""

import importlib
import os
import sys

import tensorflow as tf
import gin

# Appending vitaFlow main Path
sys.path.append(os.path.abspath('.'))

from vitaflow.engines.experiments import Experiments
from vitaflow.datasets import datasets # pylint: disable=unused-import
from vitaflow.iterators import iterators # pylint: disable=unused-import
from vitaflow.models import models # pylint: disable=unused-import

import vitaflow.utils.registry as registry
flags = tf.flags
flags.DEFINE_string("config_file", "Google gin config file", "path/to/gin_config.gin")
flags.DEFINE_string("mode", "train", "train/retrain/predict/predict_instance")
flags.DEFINE_bool("registry_help", False,
    "If True, logs the contents of the registry and exits.")
FLAGS = flags.FLAGS


def maybe_log_registry_and_exit():
    if FLAGS.registry_help:
        tf.logging.info(registry.help_string())
        sys.exit(0)


def main():
    print(' -' * 35)
    print('Running Experiment:')
    print(' -' * 35)
    experiment = Experiments(mode=FLAGS.mode)
    experiment.run(args=FLAGS)
    print(' -' * 35)


if __name__ == "__main__":
    # If we just have to print the registry, do that and exit early.
    maybe_log_registry_and_exit()

    gin.parse_config_file(FLAGS.config_file)
    main()
