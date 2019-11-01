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
Run Tensorflow VitaFlowEngine for given dataset, iterator and _model

run_experiments \
	--config=path/to/gin_config.gin
"""

import os
import sys
from absl import app
from absl import flags
from absl import logging
import warnings
# Appending vitaFlow main Path
sys.path.append(os.path.abspath('.'))
warnings.simplefilter(action='ignore', category=FutureWarning)
logging.set_verbosity(logging.DEBUG)
import tensorflow as tf
import gin

from vitaflow.serving.iservice import IServing
from vitaflow.serving.serving import get_serving

from vitaflow.engine.engine import VitaFlowEngine
from vitaflow.datasets import datasets # pylint: disable=unused-import
from vitaflow.models import models # pylint: disable=unused-import
from vitaflow.serving import serving # pylint: disable=unused-import

FLAGS = flags.FLAGS
flags.DEFINE_string("config_file", "Google gin config file", "path/to/gin_config.gin")
flags.DEFINE_string("serving_class_name", "", "")
flags.DEFINE_string("in_file_or_path", "", "")
flags.DEFINE_string("out_file_or_path", "", "")

def main(argv):
    gin.parse_config_file(FLAGS.config_file)
    print(' -' * 35)
    print('Running VitaFlowEngine Serving with config file {}:'.format(FLAGS.config_file))
    print(' -' * 35)
    serving_instance: IServing = get_serving(FLAGS.serving_class_name)
    serving_instance = serving_instance()
    serving_instance.predict(in_file_or_path=FLAGS.in_file_or_path,
                             out_file_or_path=FLAGS.out_file_or_path)
    print(' -' * 35)


if __name__ == "__main__":
    app.run(main)
