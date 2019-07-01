# Copyright 2018 The vitaFlow Authors. All Rights Reserved.
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
A class that sets up default Tensorflow operations for classifying audio
"""
import os
import tensorflow as tf
from overrides import overrides

# from vitaflow.utils.hyperparams import HParams
import gin
from vitaflow.models.internals.model_base import ModelBase

@gin.configurable
class ClassifierBase(ModelBase):
    """
    Base class for classification models
    """

    def __init__(self,
                 experiment_name,
                model_root_directory=os.path.join(os.path.expanduser("~"), "vitaFlow/", "default_model_dir"),
                name="classifier_base",
                out_dim=-1,
                learning_rate=0.001):
        ModelBase.__init__(self,
                           experiment_name=experiment_name,
                           model_root_directory=model_root_directory)
        # self._hparams = HParams(hparams, self.default_hparams())

        self._out_dim = out_dim
        self._learning_rate = learning_rate

    def __call__(self, features, labels, params, mode, config=None):
        """
        Used for the :tf_main:`model_fn <estimator/Estimator#__init__>`
        argument when constructing
        :tf_main:`tf.estimator.Estimator <estimator/Estimator>`.
        """
        return self._build(features, labels, params, mode, config=config)

    # @staticmethod
    # def default_hparams():
    #     """
    #     .. role:: python(code)
    #        :language: python
    #
    #     .. code-block:: python
    #
    #         {
    #             "experiment_name": "model_name_or_dataset_name",
    #             "model_root_directory" : os.path.join(os.path.expanduser("~"), "vitaFlow/", "default_model_dir"),
    #             "name": "classifier_base",
    #             "out_dim": -1,
    #             "learning_rate": 0.001,
    #         }
    #
    #     Here:
    #
    #     "experiment_name" : str
    #         Name of the experiment
    #     "model_root_directory" : str
    #         Model root directory to store the model data under it with model class name as folder name
    #     "name": str
    #         Name of the classifier
    #     "out_dim":
    #         Number of output labels/classes
    #     "learning_rate" : float
    #         Learning rate
    #
    #     :return:  A dictionary of hyperparameters with default values
    #     """
    #     params = ModelBase.default_hparams()
    #     params.update({
    #         "name": "classifier_base",
    #         "out_dim": -1,
    #         "learning_rate": 0.001,
    #     })
    #     return params

    def _get_loss(self, labels, logits):
        """
        Loss functions for classification problems
        Refer: https://en.wikipedia.org/wiki/Loss_functions_for_classification
        :param labels:
        :param logits:
        :return:
        """
        loss = tf.losses.softmax_cross_entropy(
            onehot_labels=labels,
            logits=logits,
            scope='softmax_cross_entropy_loss')

        loss = tf.reduce_mean(loss, name="softmax_cross_entropy_mean_loss")
        return loss

    def _build_layers(self, features, mode):
        layer1 = tf.layers.dense(inputs=features,
                                 units=self._out_dim,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(seed=42))
        layer2 = tf.layers.dense(inputs=layer1,
                                 units=self._out_dim,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(seed=42))
        logits = tf.layers.dense(inputs=layer2,
                                 units=self._out_dim,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(seed=42))
        return logits

    def _get_predicted_classes(self, logits):
        """

        :param logits:
        :return:
        """
        predicted_class = tf.argmax(logits, axis=1, name="class_output")
        tf.logging.info('predicted_class: -----> {}'.format(predicted_class))
        return predicted_class

    def _get_class_probabilities(self, logits):
        predicted_probabilities = tf.nn.softmax(logits, name="softmax_output")
        tf.logging.info('predicted_probabilities: -----> {}'.format(predicted_probabilities))
        return predicted_probabilities

    # def _get_top_k_predictions(self, logits):
    #     return tf.nn.top_k(tf.nn.softmax(logits), self._k)

    def _get_optimizer(self, loss):
        optimizer = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.train.get_global_step(),
            optimizer=tf.train.AdamOptimizer,
            learning_rate=self._learning_rate)
        return optimizer

    def _get_eval_metrics(self, labels, logits):
        label_argmax = tf.argmax(labels, 1, name='label_argmax')
        predicted_class = self._get_predicted_classes(logits=logits)

        eval_metric_ops = {
            'Accuracy': tf.metrics.accuracy(
                labels=label_argmax,
                predictions=predicted_class,
                name='accuracy'),
            'Precision': tf.metrics.precision(
                labels=label_argmax,
                predictions=predicted_class,
                name='Precision'),
            'Recall': tf.metrics.recall(
                labels=label_argmax,
                predictions=predicted_class,
                name='Recall')
        }
        return eval_metric_ops

    @overrides
    def _build(self, features, labels, params, mode, config=None):
        # Loss, training and eval operations are not needed during inference.
        loss = None
        optimizer = None
        eval_metric_ops = {}

        logits = self._build_layers(features=features, mode=mode)
        predicted_class = self._get_predicted_classes(logits=logits)
        predicted_probabilities = self._get_class_probabilities(logits=logits)
        # top_k = self._get_top_k_predictions(logits=logits)
        predictions = {
            "classes": predicted_class,
            "probabilities": predicted_probabilities,
        }

        if mode != tf.estimator.ModeKeys.PREDICT:
            # labels = tf.reshape(labels, shape=(-1, self._out_dim), name="labels")
            tf.logging.info('labels: -----> {}'.format(labels))

            loss = self._get_loss(labels=labels, logits=logits)
            optimizer = self._get_optimizer(loss)
            eval_metric_ops = self._get_eval_metrics(logits=logits, labels=labels)

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=optimizer,
            eval_metric_ops=eval_metric_ops)
