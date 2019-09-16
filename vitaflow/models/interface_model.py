# Copyright 2018 The Shabda Authors. All Rights Reserved.
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
Base class for models.
"""
import os

from vitaflow.utils.hyperparams import HParams

# pylint: disable=too-many-arguments

__all__ = [
    "IEstimatorModel"
]

class IModelBase(object):

    def __init__(self,
                 experiment_name,
                 model_root_directory,
                 dataset=None):
        self._experiment_name = experiment_name
        self._model_root_directory = model_root_directory
        self._dataset = dataset

    @property
    def model_dir(self):
        """
        Returns model directory `model_root_directory`/`experiment_name`/VanillaGAN
        :return:
        """
        return os.path.join(self._model_root_directory,
                            self._experiment_name,
                            type(self).__name__)

    def get_inputs(self):
        raise NotImplementedError("User model class must implement this routine")

    def get_outputs(self):
        raise NotImplementedError("User model class must implement this routine")

    def get_loss(self, labels=None, logits=None):
        raise NotImplementedError("User model class must implement this routine")

    def get_optimizer(self, loss=None):
        raise NotImplementedError("User model class must implement this routine")


class IEstimatorModel(IModelBase):
    """Base class inherited by all model classes.

    A model class implements interfaces that are compatible with
    :tf_main:`TF Estimator <estimator/Estimator>`. In particular,
    :meth:`_build` implements the
    :tf_main:`model_fn <estimator/Estimator#__init__>` interface; and
    :meth:`get_input_fn` is for the :attr:`input_fn` interface.

    .. document private functions
    .. automethod:: _build
    """

    def __init__(self,
                 experiment_name,
                 model_root_directory=os.path.join(os.path.expanduser("~"), "vitaFlow/", "tf_model")):
        IModelBase.__init__(self,
                            experiment_name=experiment_name,
                            model_root_directory=model_root_directory,
                            datasert=dataset)

    def __call__(self, features, labels, params, mode, config=None):
        """
        Used for the :tf_main:`model_fn <estimator/Estimator#__init__>`
        argument when constructing
        :tf_main:`tf.estimator.Estimator <estimator/Estimator>`.
        """
        return self._build(features, labels, params, mode, config=config)

    def build_layers(self, features, mode):
        raise NotImplementedError

    def get_eval_metrics(self, predictions, labels):
        raise NotImplementedError

    def _build(self, features, labels, params, mode, config=None):
        """
        Used for the :tf_main:`model_fn <estimator/Estimator#__init__>`
        argument when constructing
        :tf_main:`tf.estimator.Estimator <estimator/Estimator>`.
        """
        raise NotImplementedError


class IKerasModel(IModelBase):

    def __init__(self,
                 experiment_name,
                 model_root_directory=os.path.join(os.path.expanduser("~"), "vitaFlow/", "keras_model"),
                 dataset=None):
        IModelBase.__init__(self,
                            experiment_name=experiment_name,
                            model_root_directory=model_root_directory,
                            dataset=dataset)

    def get_callbacks(self):
        raise NotImplementedError

class ITorchModel(IModelBase):

    def __init__(self,
                 experiment_name,
                 model_root_directory=os.path.join(os.path.expanduser("~"), "vitaFlow/", "keras_model"),
                 dataset=None):
        IModelBase.__init__(self,
                            experiment_name=experiment_name,
                            model_root_directory=model_root_directory,
                            dataset=dataset)
