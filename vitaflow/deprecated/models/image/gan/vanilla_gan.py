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
Vanilla GAN
"""
import os
import math

import tensorflow as tf
from tensorflow.python.training import training_util
from tensorflow.contrib.learn import ModeKeys

from vitaflow.internal import HParams
from vitaflow.internal.features import GANFeature
from vitaflow.internal.hooks.image_grid_hook import ImageGridHook
from vitaflow.internal.hooks.run_train_ops_hook import GANTrainSteps, RunTrainOpsHook
from vitaflow.utils.print_helper import print_info
from vitaflow.internal.models import ModelBase


class VanillaGAN(ModelBase, GANFeature):
    """
    A simple GAN that works on image of shape 32x32x3 and a vector of given shape.
    """
    def __init__(self, hparams=None, data_iterator=None):
        """

        :param hparams:
        :param data_iterator:
        """
        ModelBase.__init__(self, hparams=hparams)
        GANFeature.__init__(self)

        self._haprams = HParams(hparams=hparams, default_hparams=self.default_hparams())

        self._data_iterator = data_iterator

        self.num_image_channels = 3
        self.image_size = 32

        self.gen_filter_size = self._haprams.gen_filter_size
        self.learning_rate = self._haprams.learning_rate
        self.alpha = self._haprams.alpha
        self.beta1 = self._haprams.beta1
        self.z_dim = self._haprams.z_dim

    @staticmethod
    def default_hparams():
        """
        .. role:: python(code)
           :language: python

        .. code-block:: python

            {
                "experiment_name": "model_name_or_dataset_name",
                "model_root_directory" : os.path.join(os.path.expanduser("~"), "vitaFlow/", "default_model_dir")
                "gen_filter_size" : 512,
                "learning_rate" : 0.001,
                "alpha" : 0.15,
                "beta1" : 0.4,
                "z_dim" : 30
            }

        Here:

        "experiment_name" : str
            Name of the experiment
        "model_root_directory" : str
            Model root directory to store the _model data under it with _model class name as folder name
        "gen_filter_size" : int
            Geenrator filter size
        "learning_rate" : float
            Learning Rate
        "alpha" : float
            Alpha
        "beta1" : float
            The exponential decay rate for the 1st moment in the optimizer
        "z_dim" : int
            Noise vector dimension

        :return:  A dictionary of hyperparameters with default values
        """

        params = ModelBase.default_hparams()
        params.update({
            "gen_filter_size" : 512,
            "learning_rate" : 0.001,
            "alpha" : 0.15,
            "beta1" : 0.4,
            "z_dim" : 30
        })
        return params

    def _get_sequential_train_hooks(self,
                                    generator_train_op,
                                    discriminator_train_op,
                                    train_steps=GANTrainSteps(1, 1)):
        """
        Returns a hooks function for sequential GAN training.
        :param generator_train_op:
        :param discriminator_train_op:
        :param train_steps: A `GANTrainSteps` tuple that determines how many _generator
                            and _discriminator training steps to take.
        :return: A function that takes a GANTrainOps tuple and returns a list of hooks.
        """

        generator_hook = RunTrainOpsHook(generator_train_op,
                                         train_steps.generator_train_steps)
        discriminator_hook = RunTrainOpsHook(discriminator_train_op,
                                             train_steps.discriminator_train_steps)
        return [discriminator_hook, generator_hook]

    def _discriminator(self, images, reuse=False):
        """
        Create the _discriminator network
        :param image: Tensor of input image(s)
        :param reuse: Boolean if the weights should be reused
        :return: Tuple of (tensor output of the _discriminator, tensor logits of the _discriminator)
        """

        with tf.variable_scope('_discriminator', reuse=reuse):
            # Input layer consider ?x32x32x3
            x1 = tf.layers.conv2d(images, 64, 5, strides=2, padding='same',
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.02))
            relu1 = tf.maximum(0.02 * x1, x1)
            relu1 = tf.layers.dropout(relu1, rate=0.5)
            # 16x16x64
            x2 = tf.layers.conv2d(relu1, 128, 5, strides=2, padding='same',
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.02))
            bn2 = tf.layers.batch_normalization(x2, training=True)
            relu2 = tf.maximum(0.02 * bn2, bn2)
            relu2 = tf.layers.dropout(relu2, rate=0.5)
            # 8x8x128
            x3 = tf.layers.conv2d(relu2, 256, 5, strides=2, padding='same',
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.02))
            bn3 = tf.layers.batch_normalization(x3, training=True)
            relu3 = tf.maximum(0.02 * bn3, bn3)
            relu3 = tf.layers.dropout(relu3, rate=0.5)
            # 4x4x256
            # Flatten it
            flat = tf.reshape(relu3, (-1, 4 * 4 * 256))
            logits = tf.layers.dense(flat, 1)
            #         print(logits)
            out = tf.sigmoid(logits)
            #         print('_discriminator out: ', out)

            print_info("======> _discriminator out: {}".format(out))

            return out, logits

    def _generator(self, z, out_channel_dim, is_train=True):
        """
        Create the _generator network
        :param z: Input z on dimension Z
        :param out_channel_dim: The number of channels in the output image
        :param is_train: Boolean if _generator is being used for training
        :return: The tensor output of the _generator
        """

        with tf.variable_scope('_generator', reuse=False):
            gen_filter_size = self.gen_filter_size

            x = tf.layers.batch_normalization(z)
            # First fully connected layer
            x = tf.layers.dense(x, 8 * 8 * gen_filter_size)
            # Reshape it to start the convolutional stack
            x = tf.reshape(x, (-1, 8, 8, gen_filter_size))
            x = tf.maximum(self.alpha * x, x)

            x = tf.layers.conv2d_transpose(x, gen_filter_size//2, 5, strides=1, padding='same')
            x = tf.layers.batch_normalization(x, training=is_train)
            x = tf.maximum(self.alpha * x, x)

            gen_filter_size = gen_filter_size // 4
            # 32 //  8 = srt(4)  => 2 => (8) -> 16 -> 32
            # 64 //  8 = srt(8)  => 3 => (8) -> 16 -> 32 -> 64
            # 128 // 8 = srt(16) => 4 => (8) -> 16 -> 32 -> 64 -> 128

            # Based on image size adds Conv layer with appropriate filter size
            for i in range(int(math.sqrt(self.image_size // 8))):
                gen_filter_size = gen_filter_size // 2
                x = tf.layers.conv2d_transpose(x, gen_filter_size, 5, strides=2, padding='same')
                x = tf.layers.batch_normalization(x, training=is_train)
                x = tf.maximum(self.alpha * x, x)

                print_info("======> Convolutional Layer {} out: {}".format(i, x))

            # Output layer
            logits = tf.layers.conv2d_transpose(x, out_channel_dim, 5, strides=1, padding='same')
            # HxWxNUM_CHANNELS now
            out = tf.tanh(logits)

            print_info("======>out: {}".format(out))

            return out

    def _build_layers(self, features, mode):

        input_z = features[self.FEATURE_2_NAME]  # Audio/Noise Placeholder to the _discriminator
        input_z = tf.cast(input_z, tf.float32)
        tf.logging.info("=========> {}".format(input_z))

        if mode != ModeKeys.INFER: #Training and Evaluation

            input_real = features[self.FEATURE_1_NAME]  # Placeholder for input image vectors to the _generator
            input_real = tf.cast(input_real, tf.float32)
            tf.logging.info("=========> {}".format(input_real))

            out_channel_dim = input_real.get_shape()[-1]

            # print('Generator for fake images...')
            g_model = self._generator(input_z, out_channel_dim)
            # print('Passing _discriminator with real images...')
            d_model_real, d_logits_real = self._discriminator(input_real)
            # print('Passing _discriminator with fake images...')
            d_model_fake, d_logits_fake = self._discriminator(g_model, reuse=True)
            return g_model, d_model_real, d_logits_real, d_model_fake, d_logits_fake
        else:
            sample_image = self._generator(input_z, self.num_image_channels, is_train=False)
            return sample_image


    def _get_loss(self, d_model_real, d_logits_real, d_model_fake, d_logits_fake):
        d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real, labels=tf.ones_like(d_model_real)))
        d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.zeros_like(d_model_fake)))
        g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.ones_like(d_model_fake)))

        d_loss = d_loss_real + d_loss_fake

        return d_loss, g_loss

    def _get_optimizer(self, d_loss, g_loss, learning_rate, beta1, global_step):
        """
        Get optimization operations
        :param d_loss: Discriminator loss Tensor
        :param g_loss: Generator loss Tensor
        :param learning_rate: Learning Rate Placeholder
        :param beta1: The exponential decay rate for the 1st moment in the optimizer
        :return: A tuple of (_discriminator training operation, _generator training operation)
        """

        # Get weights and bias to update
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if var.name.startswith('_discriminator')]
        g_vars = [var for var in t_vars if var.name.startswith('_generator')]

        # Optimize
        d_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1, name="d_train_opt"). \
            minimize(d_loss, var_list=d_vars, global_step=global_step)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        g_updates = [opt for opt in update_ops if opt.name.startswith('_generator')]

        with tf.control_dependencies(g_updates):
            g_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1, name="g_train_opt"). \
                minimize(g_loss, var_list=g_vars, global_step=global_step)

        # tf.logging.info("=========> {}".format(d_train_opt))
        # tf.logging.info("=========> {}".format(g_train_opt))

        return d_train_opt, g_train_opt

    def _get_eval_metrics(self, predictions, labels):
        return {}

    def _build(self, features, labels, params, mode, config=None):
        """Used for the :tf_main:`model_fn <estimator/Estimator#__init__>`
        argument when constructing
        :tf_main:`tf.estimator.Estimator <estimator/Estimator>`.

        :param features: 
        :param labels: 
        :param mode: 
        :param params: 
        :return: 
        """

        sample_image = None
        training_hooks = None

        # Loss, training and eval operations are not needed during inference.
        loss = None
        eval_metric_ops = {}

        # Create global step increment op.
        self.global_step = training_util.get_or_create_global_step()
        self.global_step_inc = self.global_step.assign_add(0)

        if mode != ModeKeys.INFER:
            g_model, d_model_real, d_logits_real, d_model_fake, d_logits_fake = self._build_layers(features=features, mode=mode)
            d_loss, g_loss = self._get_loss(d_model_real, d_logits_real, d_model_fake, d_logits_fake)
            user_log_hooks = ImageGridHook(g_model, d_loss, g_loss, self.global_step,
                                           path=os.path.join(os.path.expanduser("~"), "vitaFlow/runtime/VanillaGAN"))
            d_train_opt, g_train_opt = self._get_optimizer(d_loss,
                                                           g_loss,
                                                           self.learning_rate,
                                                           self.beta1,
                                                           self.global_step)
            loss = g_loss + d_loss
            tf.summary.scalar(name="g_loss", tensor=g_loss)
            tf.summary.scalar(name="d_loss", tensor=d_loss)

            training_hooks = self._get_sequential_train_hooks(d_train_opt, g_train_opt)
            training_hooks.append(user_log_hooks)
        else:
            sample_image = self._build_layers(features=features, mode=mode)


        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=sample_image,
            loss=loss,
            train_op=self.global_step_inc,
            eval_metric_ops=eval_metric_ops,
            training_hooks=training_hooks
        )
