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
BEGAN
"""
import os
import math

import tensorflow as tf
from tensorflow.python.training import training_util
from tensorflow.contrib.learn import ModeKeys

from vitaflow.internal import HParams
from vitaflow.internal.features import GANFeature
from vitaflow.internal.hooks.image_grid_hook import ImageGridHook
from vitaflow.utils.print_helper import print_info, print_error
from vitaflow.internal.models import ModelBase
from vitaflow.internal.hooks.run_train_ops_hook import GANTrainSteps, RunTrainOpsHook


class BEGAN(ModelBase, GANFeature):
    """
    Paper:
        - https://arxiv.org/abs/1703.10717
    Git:
        - https://github.com/hwalsuklee/tensorflow-generative-model-collections/blob/master/BEGAN.py
        - https://github.com/carpedm20/BEGAN-tensorflow
    """

    def __init__(self, hparams=None, data_iterator=None):
        ModelBase.__init__(self, hparams=hparams)
        GANFeature.__init__(self)
        self._hparams = HParams(hparams=hparams, default_hparams=self.default_hparams())

        self._data_iterator = data_iterator

        self.num_image_channels = 3
        self.image_size = 32

        self.gen_filter_size = self._hparams.gen_filter_size
        self.learning_rate = self._hparams.learning_rate
        self.z_dim = self._hparams.z_dim


        # BEGAN Parameter
        self.gamma = self._hparams.gamma
        self.lamda = self._hparams.lamda

        self.beta1 = self._hparams.beta1
        self.alpha = self._hparams.alpha

    @staticmethod
    def default_hparams():
        """
        .. role:: python(code)
           :language: python

        .. code-block:: python

            {
                "experiment_name": "model_name_or_dataset_name",
                "model_root_directory" : os.path.join(os.path.expanduser("~"), "vitaFlow/", "default_model_dir")
                "gen_filter_size" : 1024,
                "learning_rate" : 0.001,
                "gamma" : 0.75,
                "lamda" : 0.001,
                "alpha" : 0.15,
                "beta1" : 0.4,
                "z_dim" : 30
            }

        Here:

        "experiment_name" : str
            Name of the experiment
        "model_root_directory" : str
            Model root directory to store the model data under it with model class name as folder name
        "gen_filter_size" : int
            Geenrator filter size
        "learning_rate" : float
            Learning Rate
        "gamma" : float
            Gamma
        "lamda" : float
            lamda
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
            "gen_filter_size" : 1024,
            "learning_rate" : 0.001,
            "gamma" : 0.75,
             "lamda" : 0.001,
            "alpha" : 0.15,
            "beta1" : 0.4,
            "z_dim" : 30
        })
        return params

    def get_sequential_train_hooks(self,
                                   generator_train_op,
                                   discriminator_train_op,
                                   train_steps=GANTrainSteps(1, 1)):
        """Returns a hooks function for sequential GAN training.

        Args:
          train_steps: A `GANTrainSteps` tuple that determines how many namespace_generator
            and namespace_discriminator training steps to take.

        Returns:
          A function that takes a GANTrainOps tuple and returns a list of hooks.
        """
        # print_info(generator_train_op)
        # print_info(discriminator_train_op)

        generator_hook = RunTrainOpsHook(generator_train_op,
                                         train_steps.generator_train_steps)
        discriminator_hook = RunTrainOpsHook(discriminator_train_op,
                                             train_steps.discriminator_train_steps)
        return [discriminator_hook, generator_hook]


    def discriminator(self, x, out_channel_dim, is_training=True, reuse=False):
        # It must be Auto-Encoder style architecture
        # Architecture : (64)4c2s-FC32_BR-FC64*14*14_BR-(1)4dc2s_S
        with tf.variable_scope("namespace_discriminator", reuse=reuse):
            # net = tf.nn.relu(conv2d(x, 64, 4, 4, 2, 2, name='d_conv1'))
            net = tf.layers.conv2d(x, 64, 4, strides=2, padding='same',
                                   kernel_initializer=tf.random_normal_initializer(stddev=0.02),
                                   name='d_conv1')
            net = tf.nn.relu(net)

            tf.logging.info("======> net: {}".format(net))
            print_error("net1: {} ".format(net))

            size = (self.image_size//2)


            net = tf.reshape(net, [self._data_iterator.batch_size, size*size*64])

            # code = tf.nn.relu(bn(linear(net, 32, scope='d_fc6'), is_training=is_training, scope='d_bn6'))
            code = tf.contrib.layers.fully_connected(inputs=net, num_outputs=32, scope="d_fc6")
            code = tf.contrib.layers.batch_norm(code,
                                                decay=0.9,
                                                updates_collections=None,
                                                epsilon=1e-5,
                                                scale=True,
                                                is_training=is_training,
                                                scope='d_bn6')
            code = tf.nn.relu(code)

            print_error("code: {} ".format(code))
            # net = tf.nn.relu(bn(linear(code, 64 * 14 * 14, scope='d_fc3'), is_training=is_training, scope='d_bn3'))
            size = (self.image_size // 2)
            net =  tf.contrib.layers.fully_connected(inputs=code, num_outputs=64 * size * size, scope="d_fc3")

            net = tf.contrib.layers.batch_norm(net,
                                               decay=0.9,
                                               updates_collections=None,
                                               epsilon=1e-5,
                                               scale=True,
                                               is_training=is_training,
                                               scope='d_bn3')
            print_error("net: {} ".format(net))
            print_error(net)

            size = (self.image_size // 2)
            net = tf.reshape(net, [self._data_iterator.batch_size, size, size, 64])
            print_error(net)

            # out = tf.nn.sigmoid(deconv2d(net, [self.gan_config.batch_size, 28, 28, 1], 4, 4, 2, 2, name='d_dc5'))
            net =  tf.layers.conv2d_transpose(net, out_channel_dim, 4, strides=2, padding='same', name='d_dc5')
            out = tf.nn.sigmoid(net)

            print_info("==================================")
            print_info(out)
            print_info(x)
            # recon loss
            recon_error = tf.sqrt(2 * tf.nn.l2_loss(out - x)) / self._data_iterator.batch_size
            print_info("==================================")
            print_error(recon_error)

            return out, recon_error, code


    def generator(self, z, out_channel_dim, is_training=True, reuse=False):
        """
        Create the namespace_generator network
        :param z: Input z
        :param out_channel_dim: The number of channels in the output image
        :param is_training: Boolean if namespace_generator is being used for training
        :return: The tensor output of the namespace_generator
        """

        with tf.variable_scope('namespace_generator', reuse=not is_training): #reuse if it not training phase
            filter_size = 512

            # First fully connected layer
            x = tf.layers.dense(z, 8 * 8 * filter_size)
            # Reshape it to start the convolutional stack
            x = tf.reshape(x, (-1, 8, 8, filter_size))
            x = tf.maximum(self.alpha * x, x)

            x = tf.layers.conv2d_transpose(x, filter_size // 2, 5, strides=1, padding='same')
            x = tf.layers.batch_normalization(x, training=is_training)
            x = tf.maximum(self.alpha * x, x)

            filter_size = filter_size // 4
            # 32 //  8 = srt(4)  => 2 => (8) -> 16 -> 32
            # 64 //  8 = srt(8)  => 3 => (8) -> 16 -> 32 -> 64
            # 128 // 8 = srt(16) => 4 => (8) -> 16 -> 32 -> 64 -> 128

            for i in range(int(math.sqrt(self.image_size // 8))):
                filter_size = filter_size // 2
                x = tf.layers.conv2d_transpose(x, filter_size, 5, strides=2, padding='same')
                x = tf.layers.batch_normalization(x, training=is_training)
                x = tf.maximum(self.alpha * x, x)

                print_info("======>out: {}".format(x))

            # Output layer
            logits = tf.layers.conv2d_transpose(x, out_channel_dim, 5, strides=1, padding='same')
            # 28x28x3 now
            #         print(logits)3
            out = tf.tanh(logits)

            print_info("======>out: {}".format(out))

            return out


    def _build_layers(self, features, mode):

        is_training = mode != ModeKeys.INFER


        input_z = features[self.FEATURE_2_NAME]  # Audio/Noise Placeholder to the namespace_discriminator
        input_z = tf.cast(input_z, tf.float32)
        tf.logging.info("=========> {}".format(input_z))

        if mode != ModeKeys.INFER: #Training and Evaluation

            input_real = features[self.FEATURE_1_NAME]  # Placeholder for input image vectors to the namespace_generator
            input_real = tf.cast(input_real, tf.float32)
            tf.logging.info("=========> {}".format(input_real))

            out_channel_dim = input_real.get_shape()[-1]

            # output of D for real images
            D_real_img, D_real_err, D_real_code = self.discriminator(input_real, out_channel_dim=out_channel_dim,
                                                                     is_training=is_training, reuse=False)

            # output of D for fake images
            G = self.generator(input_z,  out_channel_dim=out_channel_dim, is_training=is_training, reuse=False)
            D_fake_img, D_fake_err, D_fake_code = self.discriminator(G, out_channel_dim=out_channel_dim, is_training=is_training, reuse=True)

            return G, D_real_img, D_real_err, D_real_code, D_fake_img, D_fake_err, D_fake_code
        else:
            sample_image = self.generator(input_z, self.num_image_channels)
            return sample_image


    def _get_loss(self, D_real_img, D_real_err, D_real_code, D_fake_img, D_fake_err, D_fake_code):
        # get loss for namespace_discriminator
        d_loss = D_real_err - self.k * D_fake_err

        # get loss for namespace_generator
        g_loss = D_fake_err

        # convergence metric
        M = D_real_err + tf.abs(self.gamma * D_real_err - D_fake_err)

        # operation for updating k
        update_k = self.k.assign(
            tf.clip_by_value(self.k + self.lamda * (self.gamma * D_real_err - D_fake_err), 0, 1))

        return d_loss, g_loss, update_k

    def _get_optimizer(self, d_loss, g_loss, global_step):
        """
        Get optimization operations
        :param d_loss: Discriminator loss Tensor
        :param g_loss: Generator loss Tensor
        :param learning_rate: Learning Rate Placeholder
        :param beta1: The exponential decay rate for the 1st moment in the optimizer
        :return: A tuple of (namespace_discriminator training operation, namespace_generator training operation)
        """

        # divide trainable variables into a group for D and a group for G
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if var.name.startswith('namespace_discriminator')]
        g_vars = [var for var in t_vars if var.name.startswith('namespace_generator')]

        # optimizers
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            d_optim = tf.train.AdamOptimizer(self.learning_rate,
                                             beta1=self.beta1) \
                .minimize(d_loss, var_list=d_vars,global_step=global_step)
            g_optim = tf.train.AdamOptimizer(self.learning_rate * 5, beta1=self.beta1) \
                .minimize(g_loss, var_list=g_vars,global_step=global_step)

        # tf.logging.info("=========> {}".format(d_train_opt))
        # tf.logging.info("=========> {}".format(g_train_opt))

        return d_optim, g_optim

    def _get_eval_metrics(self, predictions, labels):
        raise {}

    def _build(self, features, labels, params, mode, config=None):
        """

        :param features: 
        :param labels: 
        :param mode: 
        :param params: 
        :return: 
        """

        sample_image = None
        training_hooks = None

        is_training = mode != ModeKeys.INFER

        # Create global step increment op.
        self.global_step = training_util.get_or_create_global_step()
        self.global_step_inc = self.global_step.assign_add(0)

        """ BEGAN variable """
        self.k = tf.Variable(0., trainable=False)

        # Loss, training and eval operations are not needed during inference.
        loss = None
        train_op = None
        eval_metric_ops = {}

        if mode != ModeKeys.INFER:
            G, D_real_img, D_real_err, D_real_code, D_fake_img, D_fake_err, D_fake_code = \
                self._build_layers(features=features, mode=mode)
            d_loss, g_loss, update_k = self._get_loss(D_real_img, D_real_err, D_real_code, D_fake_img, D_fake_err, D_fake_code)
            print_hooks = ImageGridHook(G, d_loss, g_loss, self.global_step,
                                        path=os.path.join(os.path.expanduser("~"), "vitaFlow/runtime/BEGAN"))

            d_train_opt, g_train_opt = self._get_optimizer(d_loss, g_loss, self.global_step)
            loss = g_loss + d_loss
            tf.summary.scalar(name="g_loss", tensor=g_loss)
            tf.summary.scalar(name="d_loss", tensor=d_loss)

            training_hooks = self.get_sequential_train_hooks(d_train_opt, g_train_opt)
            update_k_hook = RunTrainOpsHook(update_k, 1)
            training_hooks.append(print_hooks)
            training_hooks.append(update_k_hook)

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
