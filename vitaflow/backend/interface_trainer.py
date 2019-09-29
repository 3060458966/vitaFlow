# Copyright 2019 The Shabda Authors. All Rights Reserved.
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

from vitaflow.datasets.interface_dataset import IDataset

class TrainerBase(object):
    def __init__(self,
                 experiment_name,
                 model,
                 dataset,
                 max_train_steps,
                 validation_interval_steps,
                 stored_model):
        """
        Base class training engine
        :param experiment_name: Name of the experiment
        :param max_train_steps: Maximum number of training steps for current experimentation
                                (step = number of samples / batch size)
        :param validation_interval_steps: Number of training steps before running validation
        :param stored_model: Previously trained model path
        :param model: Respective model class that the inherited trainer can handle
        :param dataset: Dataset class
        """
        self._experiment_name = experiment_name
        self._max_train_steps = max_train_steps
        self._validation_interval_steps = validation_interval_steps
        self._stored_model = stored_model
        self._model = model
        self._dataset = dataset

    def train(self, max_steps=None, num_epochs=None):
        raise NotImplementedError

    def evaluate(self, steps=None, checkpoint_path=None):
        raise NotImplementedError

    def train_and_evaluate(self, max_train_steps=None, eval_steps=None, num_epochs=None):
        raise NotImplementedError

    def predict_directory(self, in_path, out_path):
        raise NotImplementedError

    def predict_file(self, in_path, out_path):
        raise NotImplementedError
