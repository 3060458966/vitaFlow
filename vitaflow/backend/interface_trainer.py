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
import os

from vitaflow.datasets.interface_dataset import IDataset

class TrainerBase(object):
    def __init__(self,
                 model,
                 dataset,
                 model_store_path):
        """
        Base class training engine
        :param experiment_name: Name of the experiment
        :param model_store_path: Previously trained _model path
        :param model: Respective _model class that the inherited trainer can handle
        :param dataset: Dataset class
        """
        self._model = model
        self._dataset = dataset
        self._model_store_path = model_store_path

    def train(self,
              max_steps=None,
              num_epochs=None,
              store_model_epoch_interval=None,
              store_model_steps_interval=None):
        raise NotImplementedError

    def evaluate(self, steps=None, checkpoint_path=None):
        raise NotImplementedError

    def train_and_evaluate(self, max_train_steps=None, eval_steps=None, num_epochs=None):
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError


