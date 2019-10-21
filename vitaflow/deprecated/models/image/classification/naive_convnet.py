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
Naive Convolutional Network
"""

import os
import gin
from overrides import overrides
import tensorflow as tf

# from vitaflow.internal import HParams
from vitaflow.internal.features import ImageFeature
from vitaflow.internal.models import ClassifierBase
from vitaflow.utils.print_helper import print_info

@gin.configurable
class NaiveConvNet(ClassifierBase, ImageFeature):
    def __init__(self,
                 experiment_name,
                 name="NaiveConvNet",
                 model_root_directory=os.path.expanduser("~") + "/vitaFlow/",
                 out_dim=-1,
                 learning_rate=0.001,
                 keep_probability=0.5,
                 data_iterator=None):
        ClassifierBase.__init__(self,
                                experiment_name=experiment_name,
                                model_root_directory=model_root_directory,
                                experiment_name=name,
                                out_dim=out_dim,
                                learning_rate=learning_rate)
        ImageFeature.__init__(self)
        # self._hparams = HParams(hparams, self.default_hparams())

        self._data_iterator = data_iterator
        self._keep_prob = keep_probability

        self._conv_num_outputs = 32 # TODO
        self._conv_ksize = (5, 5)
        self._conv_strides = (1, 1)
        self._pool_ksize = (2, 2)
        self._pool_strides = (2, 2)

        self._num_outputs = 10 #number of classes # TODO

        print_info("NaiveConvNet initialized")

    @staticmethod
    def default_hparams():
        """
        .. role:: python(code)
           :language: python

        .. code-block:: python

            {
                "model_root_directory" : os.path.expanduser("~") + "/vitaFlow/",
                "experiment_name" : "experiment_name",
                "name": "NaiveConvNet",
                "out_dim": -1,
                "learning_rate": 0.001
                # hyper parameters
                "keep_probability": 0.5,
            }

        Here:
        "experiment_name" : str
            Name of the experiment
        "model_root_directory" : str
            Model root directory to store the _model data under it with _model class name as folder name
        "name": str
            Name of the classifier
        "out_dim":
            Number of output labels/classes
        "learning_rate" : float
            Learning rate
        "keep_probability" : float
            Drop out layer `keep` probability value

        :return: A dictionary of hyper parameters with default values
        """
        hparams = ClassifierBase.default_hparams()
        hparams = hparams.update({
            "name": "NaiveConvNet",
            # hyper parameters
            "keep_probability": 0.5,
        })
        return hparams

    def new_weights(self, shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

    def new_biases(self, length):
        #equivalent to y intercept
        #constant value carried over across matrix math
        return tf.Variable(tf.constant(0.05, shape=[length]))

    def conv2d_maxpool(self, x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides):
        """
        Apply convolution then max pooling to x_tensor
        :param x_tensor: TensorFlow Tensor
        :param conv_num_outputs: Number of outputs for the convolutional layer
        :param conv_ksize: kernal size 2-D Tuple for the convolutional layer
        :param conv_strides: Stride 2-D Tuple for convolution
        :param pool_ksize: kernal size 2-D Tuple for pool
        :param pool_strides: Stride 2-D Tuple for pool
        : return: A tensor that represents convolution and max pooling of x_tensor
        """
        print("Creating conv2d_maxpool layer")
        num_input_channel = x_tensor.get_shape()
        assert(len(num_input_channel) == 4)
        shape = [conv_ksize[0], conv_ksize[1], num_input_channel[3].value, conv_num_outputs]
        #     print(shape)
        #     print(type(shape))
        weights = self.new_weights(shape=shape)
        biases = self.new_biases(length=conv_num_outputs)

        layer = tf.nn.conv2d(input=x_tensor,
                             filter=weights,
                             strides=[1, conv_strides[0], conv_strides[1], 1],
                             padding='SAME')
        print("   ---> conv layer: ", layer.get_shape())
        layer += biases

        layer = tf.nn.relu(layer)
        print("   ---> relu layer: ", layer.get_shape())

        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, pool_ksize[0], pool_ksize[1], 1],
                               strides=[1, pool_strides[0], pool_strides[1], 1],
                               padding='SAME')
        print("   ---> max pool layer: ", layer.get_shape())

        return layer

    def flatten(self, x_tensor):
        """
        Flatten x_tensor to (Batch Size, Flattened Image Size)
        : x_tensor: A tensor of size (Batch Size, ...), where ... are the image dimensions.
        : return: A tensor of size (Batch Size, Flattened Image Size).
        """
        layer_shape = x_tensor.get_shape()
        num_features = layer_shape[1:4].num_elements()
        layer_flat = tf.reshape(x_tensor, [-1, num_features])

        return layer_flat

    def fully_conn(self, x_tensor, num_outputs):
        """
        Apply a fully connected layer to x_tensor using weight and bias
        : x_tensor: A 2-D tensor where the first dimension is batch size.
        : num_outputs: The number of output that the new tensor should be.
        : return: A 2-D tensor where the second dimension is num_outputs.
        """
        x_shape = x_tensor.get_shape()
        assert(len(x_shape) == 2)
        num_inputs = x_shape[1].value
        # Create new weights and biases.
        weights = self.new_weights(shape=[num_inputs, num_outputs])
        biases = self.new_biases(length=num_outputs)

        # Calculate the layer as the matrix multiplication of
        # the input and weights, and then add the bias-values.
        layer = tf.matmul(x_tensor, weights) + biases
        layer = tf.nn.relu(layer)
        return layer


    def output(self, x_tensor, num_outputs):
        """
        Apply a output layer to x_tensor using weight and bias
        : x_tensor: A 2-D tensor where the first dimension is batch size.
        : num_outputs: The number of output that the new tensor should be.
        : return: A 2-D tensor where the second dimension is num_outputs.
        """
        x_shape = x_tensor.get_shape()
        assert(len(x_shape) == 2)
        num_inputs = x_shape[1].value
        # Create new weights and biases.
        weights = self.new_weights(shape=[num_inputs, num_outputs])
        biases = self.new_biases(length=num_outputs)

        # Calculate the layer as the matrix multiplication of
        # the input and weights, and then add the bias-values.
        layer = tf.matmul(x_tensor, weights) + biases

        return layer

    def _get_loss(self, labels, logits, **kwargs):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    def _build_layers(self, features, mode):
        print("x: ", features.get_shape())
        # TODO: Apply 1, 2, or 3 Convolution and Max Pool layers
        #    Play around with different number of outputs, kernel size and stride
        # Function Definition from Above:
        #    conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides)
        layer = self.conv2d_maxpool(features, 16, self._conv_ksize, self._conv_strides,
                                    self._pool_ksize, self._pool_strides)
        print("conv1 layer: ", layer.get_shape())
        layer = tf.nn.dropout(layer, self._keep_prob)
        print("conv1_dropout layer: ", layer.get_shape())
        layer = self.conv2d_maxpool(layer, 32, self._conv_ksize, self._conv_strides,
                                    self._pool_ksize, self._pool_strides)
        print("conv2 layer: ", layer.get_shape())


        # Function Definition from Above:
        #   flatten(x_tensor)
        layer = self.flatten(layer)
        print("flatten layer: ", layer.get_shape())
        layer = tf.nn.dropout(layer, self._keep_prob)
        print("flatten_dropout layer : ", layer.get_shape())

        #    Play around with different number of outputs
        # Function Definition from Above:
        #   fully_conn(x_tensor, num_outputs)
        layer = self.fully_conn(layer, 1024)
        print("fully_connected1 layer: ", layer.get_shape())
        #     layer = fully_conn(layer, 512)
        #     print("fully_connected2 layer: ", layer.get_shape())
        #     layer = fully_conn(layer, num_outputs)

        #    Set this to the number of classes
        # Function Definition from Above:
        #   output(x_tensor, num_outputs)
        layer = self.output(layer, self._num_outputs)
        print("output layer: ", layer.get_shape(), "\n")
        return layer


    @overrides
    def _build(self, features, labels, params, mode, config=None):

        images = features[self.FEATURE_NAME]

        shape = images.get_shape()
        assert(len(shape) == 4)
        batch = shape[0].value
        rows = shape[1].value
        cols = shape[2].value
        channel = shape[3].value

        print_info("{} {} {}".format(batch, rows, cols))

        # Loss, training and eval operations are not needed during inference.
        loss = None
        optimizer = None
        eval_metric_ops = {}

        logits = self._build_layers(features=images, mode=mode)
        predicted_class = self._get_predicted_classes(logits=logits)
        predicted_probabilities = self._get_class_probabilities(logits=logits)
        # top_k = self._get_top_k_predictions(logits=logits)
        predictions = {
            "classes": predicted_class,
            "probabilities": predicted_probabilities,
            "logits" : logits
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

