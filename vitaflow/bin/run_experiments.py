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

"""Run Tensorflow Experiments for given dataset, iterator and model

run_experiments \
	--mode=train \
	--config_python_file=path/to/config.py
"""

import importlib
import os
import sys

import tensorflow as tf
# from memory_profiler import profile

# Appending vitaFlow main Path
sys.path.append(os.path.abspath('.'))

from vitaflow.engines.experiments import Experiments

# TODO: Use ArgParse
# tf.args works just like argparse
flags = tf.flags
flags.DEFINE_string("config_python_file", "config_python_file", "The config to use.")
flags.DEFINE_string("mode", "train", "train/retrain/predict/predict_instance")
flags.DEFINE_string("test_file_path", "", "")

FLAGS = flags.FLAGS
config = FLAGS.config_python_file.replace("/", ".")
config = config.replace(".py", "")
config = importlib.import_module(config)

#@profile
def main():
    print(' -' * 35)
    print('Running Experiment:',config.experiment_name)
    print(' -' * 35)
    import spacy

    try:
        spacy.load('en_core_web_md')
    except OSError:
        download_en_core_web_md = 'python -m spacy download en_core_web_md'
        import subprocess

        subprocess.call(download_en_core_web_md.split())
        print('OSError: Not able to find Spacy Package - Downloading "en_core_web_md" !')
        print('OSError: Run following command & re-run the experiment')
        print(' -' * 35)
        print('Downloading missing package(command: {})'.format(download_en_core_web_md))
        print(' -' * 35)

    if FLAGS.mode != "train":
        config.experiments['clear_model_data'] = False #make sure we are not deleting the model accidentally

    experiment = Experiments(hparams=config.experiments, mode=FLAGS.mode)
    experiment.run(args=FLAGS)
    print(' -' * 35)


if __name__ == "__main__":
    main()
