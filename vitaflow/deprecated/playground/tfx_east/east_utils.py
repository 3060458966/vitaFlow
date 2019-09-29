# Copyright 2019 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Python source file include east_airflow_demo pipeline functions and necesasry utils.

For a TFX pipeline to successfully run, a preprocessing_fn and a
_build_estimator function needs to be provided.  This file contains both.
"""

from __future__ import division
from __future__ import print_function

import os  # pylint: disable=unused-import

import numpy as np
import tensorflow as tf  # pylint: disable=unused-import
import tensorflow_model_analysis as tfma  # Step 5
from tensorflow.contrib import slim
from tensorflow_transform.beam.tft_beam_io import transform_fn_io  # Step 4
from tensorflow_transform.saved import saved_transform_io  # Step 4
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import dataset_schema
from tensorflow_transform.tf_metadata import metadata_io  # Step 4
from tensorflow_transform.tf_metadata import schema_utils  # Step 4
from tfx_east.print_helper import *
from tfx_east import resnet_v1


# Step 4 START --------------------------
def _transformed_name(key):
    return key + '_xf'


def _transformed_names(keys):
    return [_transformed_name(key) for key in keys]


# # Tf.Transform considers these features as "raw"
def _get_raw_feature_spec(schema):
    RAW_DATA_FEATURE_SPEC = {
        _transformed_name('images'): tf.FixedLenFeature([512 * 512 * 3], tf.float32),
        _transformed_name('score_maps'): tf.FixedLenFeature([128 * 128 * 1], tf.float32),
        _transformed_name('geo_maps'): tf.FixedLenFeature([128 * 128 * 5], tf.float32),
        _transformed_name('training_masks'): tf.FixedLenFeature([128 * 128 * 1], tf.float32),
    }
    return schema_utils.schema_as_feature_spec(RAW_DATA_FEATURE_SPEC).feature_spec


def _gzip_reader_fn():
    """Small utility returning a record reader that can read gzip'ed files."""
    return tf.TFRecordReader(
        options=tf.python_io.TFRecordOptions(
            compression_type=tf.python_io.TFRecordCompressionType.GZIP))


RAW_DATA_FEATURE_SPEC = {
    _transformed_name('images'): tf.FixedLenFeature([512 * 512 * 3], tf.float32),
    _transformed_name('score_maps'): tf.FixedLenFeature([128 * 128 * 1], tf.float32),
    _transformed_name('geo_maps'): tf.FixedLenFeature([128 * 128 * 5], tf.float32),
    _transformed_name('training_masks'): tf.FixedLenFeature([128 * 128 * 1], tf.float32),
}

RAW_DATA_METADATA = dataset_metadata.DatasetMetadata(
    dataset_schema.from_feature_spec(RAW_DATA_FEATURE_SPEC))


def preprocessing_fn(inputs):
    """tf.transform's callback function for preprocessing inputs.

    Args:
      inputs: map from feature keys to raw not-yet-transformed features.

    Returns:
      Map from string feature key to transformed feature operations.
    """
    print_info("preprocessing_fn")
    print_info(inputs)

    _outputs = {}

    _DENSE_FLOAT_FEATURE_KEYS = ["images", "training_masks", "geo_maps", "score_maps"]

    _outputs[_transformed_name("images")] = tf.reshape(tf.cast(tf.sparse.to_dense(inputs['images']),
                                                               tf.float32), shape=[-1,512, 512, 3])
    _outputs[_transformed_name('score_maps')] = tf.reshape(tf.cast(tf.sparse.to_dense(inputs['score_maps']),
                                                                   tf.float32), shape=[-1,128, 128, 1])
    _outputs[_transformed_name('geo_maps')] = tf.reshape(tf.cast(tf.sparse.to_dense(inputs['geo_maps']),
                                                                 tf.float32), shape=[-1,128, 128, 5])
    _outputs[_transformed_name('training_masks')] = tf.reshape(tf.cast(tf.sparse.to_dense(inputs['training_masks']),
                                                                       tf.float32), shape=[-1,128, 128, 1])

    return _outputs
# Step 4 END --------------------------

def _build_estimator(config):
    """Build an estimator for predicting the tipping behavior of east_airflow_demo riders."""

    _model = EASTModel()
    return tf.estimator.Estimator(
        model_fn=_model, config=config, params=None)

    # return tf.estimator.DNNLinearCombinedClassifier(
    #     config=config,
    #     linear_feature_columns=categorical_columns,
    #     dnn_feature_columns=real_valued_columns,
    #     dnn_hidden_units=hidden_units or [100, 70, 50, 25],
    #     warm_start_from=warm_start_from)


def _example_serving_receiver_fn(transform_output, schema):
    """Build the serving in inputs.

    Args:
      transform_output: directory in which the tf-transform model was written
        during the preprocessing step.
      schema: the schema of the input data.

    Returns:
      Tensorflow graph which parses examples, applying tf-transform to them.
    """
    raw_feature_spec = _get_raw_feature_spec(schema)
    raw_feature_spec.pop("training_masks")
    raw_feature_spec.pop("geo_maps")
    raw_feature_spec.pop("score_maps")

    raw_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
        raw_feature_spec, default_batch_size=None)
    serving_input_receiver = raw_input_fn()

    _, transformed_features = (
        saved_transform_io.partially_apply_saved_transform(
            os.path.join(transform_output, transform_fn_io.TRANSFORM_FN_DIR),
            serving_input_receiver.features))

    return tf.estimator.export.ServingInputReceiver(
        transformed_features, serving_input_receiver.receiver_tensors)


def _eval_input_receiver_fn(transform_output, schema):
    """Build everything needed for the tf-model-analysis to run the model.

    Args:
      transform_output: directory in which the tf-transform model was written
        during the preprocessing step.
      schema: the schema of the input data.

    Returns:
      EvalInputReceiver function, which contains:
        - Tensorflow graph which parses raw untransformed features, applies the
          tf-transform preprocessing operators.
        - Set of raw, untransformed features.
        - Label against which predictions will be compared.
    """
    # Notice that the inputs are raw features, not transformed features here.
    raw_feature_spec = _get_raw_feature_spec(schema)

    serialized_tf_example = tf.placeholder(
        dtype=tf.string, shape=[None], name='input_example_tensor')

    # Add a parse_example operator to the tensorflow graph, which will parse
    # raw, untransformed, tf examples.
    features = tf.parse_example(serialized_tf_example, raw_feature_spec)

    # Now that we have our raw examples, process them through the tf-transform
    # function computed during the preprocessing step.
    _, transformed_features = (
        saved_transform_io.partially_apply_saved_transform(
            os.path.join(transform_output, transform_fn_io.TRANSFORM_FN_DIR),
            features))

    # The key name MUST be 'examples'.
    receiver_tensors = {'examples': serialized_tf_example}

    # NOTE: Model is driven by transformed features (since training works on the
    # materialized output of TFT, but slicing will happen on raw features.
    features.update(transformed_features)

    return tfma.export.EvalInputReceiver(
        features=features,
        receiver_tensors=receiver_tensors,
        labels=transformed_features[_transformed_name("training_masks"),
                                    _transformed_name("geo_maps"),_transformed_name("score_maps")])


def _input_fn(filenames, transform_output, batch_size=200):
    """Generates features and labels for training or evaluation.

    Args:
      filenames: [str] list of CSV files to read data from.
      transform_output: directory in which the tf-transform model was written
        during the preprocessing step.
      batch_size: int First dimension size of the Tensors returned by input_fn

    Returns:
      A (features, indices) tuple where features is a dictionary of
        Tensors, and indices is a single Tensor of label indices.
    """
    metadata_dir = os.path.join(transform_output,
                                transform_fn_io.TRANSFORMED_METADATA_DIR)
    transformed_metadata = metadata_io.read_metadata(metadata_dir)
    transformed_feature_spec = transformed_metadata.schema.as_feature_spec()
    #
    transformed_features = tf.contrib.learn.io.read_batch_features(
        filenames, batch_size, transformed_feature_spec, reader=_gzip_reader_fn)
    #
    # # We pop the label because we do not want to use it as a feature while we're
    # # training.,"training_masks","geo_maps","score_maps"
    # return transformed_features, transformed_features[
    #     _transformed_name("images")]

    # """Input function for training and eval."""
    # dataset = tf.contrib.data.make_batched_features_dataset(file_pattern = filenames,
    #                                                         batch_size = batch_size,
    #                                                         features = transformed_feature_spec,
    #                                                         reader = tf.data.TFRecordDataset,
    #                                                         shuffle = True)
    #
    # dataset = dataset.map(decode)
    # transformed_features = dataset.make_one_shot_iterator().get_next()

    # Extract features and label from the transformed tensors.
    tf.logging.info("<->")
    tf.logging.info(transformed_features.keys())
    return transformed_features,{k: v for k, v in transformed_features.items() if k in [_transformed_name("training_masks"),_transformed_name("geo_maps"),_transformed_name("score_maps")] }


# TFX will call this function
def trainer_fn(hparams, schema):
    """Build the estimator using the high level API.

    Args:
      hparams: Holds hyperparameters used to train the model as name/value pairs.
      schema: Holds the schema of the training examples.

    Returns:
      A dict of the following:
        - estimator: The estimator that will be used for training and eval.
        - train_spec: Spec for training.
        - eval_spec: Spec for eval.
        - eval_input_receiver_fn: Input function for eval.
    """
    # Number of nodes in the first layer of the DNN
    # first_dnn_layer_size = 100
    # num_dnn_layers = 4
    # dnn_decay_factor = 0.7

    train_batch_size = 4
    eval_batch_size = 4

    train_input_fn = lambda: _input_fn(  # pylint: disable=g-long-lambda
        hparams.train_files,
        hparams.transform_output,
        batch_size=train_batch_size)

    eval_input_fn = lambda: _input_fn(  # pylint: disable=g-long-lambda
        hparams.eval_files,
        hparams.transform_output,
        batch_size=eval_batch_size)

    train_spec = tf.estimator.TrainSpec(  # pylint: disable=g-long-lambda
        train_input_fn,
        max_steps=hparams.train_steps)

    serving_receiver_fn = lambda: _example_serving_receiver_fn(  # pylint: disable=g-long-lambda
        hparams.transform_output, schema)

    exporter = tf.estimator.FinalExporter('east_airflow_demo', serving_receiver_fn)
    eval_spec = tf.estimator.EvalSpec(
        eval_input_fn,
        steps=hparams.eval_steps,
        exporters=[exporter],
        name='east_airflow_demo-eval')

    run_config = tf.estimator.RunConfig(
        save_checkpoints_steps=999, keep_checkpoint_max=1)

    run_config = run_config.replace(model_dir=hparams.serving_model_dir)

    estimator = _build_estimator(
        # Construct layers sizes with exponetial decay
        # hidden_units=[
        #     max(2, int(first_dnn_layer_size * dnn_decay_factor**i))
        #     for i in range(num_dnn_layers)
        # ],
        # warm_start_from=hparams.warm_start_from
        config=run_config
    )

    # Create an input receiver for TFMA processing
    receiver_fn = lambda: _eval_input_receiver_fn(  # pylint: disable=g-long-lambda
        hparams.transform_output, schema)

    return {
        'estimator': estimator,
        'train_spec': train_spec,
        'eval_spec': eval_spec,
        'eval_input_receiver_fn': receiver_fn
    }
#############################################################
def unpool(inputs):
    return tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1] * 2, tf.shape(inputs)[2] * 2])


def mean_image_subtraction(images, means=[123.68, 116.78, 103.94]):
    """
    image normalization
    :param images:
    :param means:
    :return:
    """

    num_channels = images.get_shape().as_list()[-1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')
    channels = tf.split(axis=3, num_or_size_splits=num_channels, value=images)
    for i in range(num_channels):
        channels[i] = channels[i] - tf.convert_to_tensor(means[i])
    return tf.concat(axis=3, values=channels)


def model(images, text_scale=512, weight_decay=1e-5, is_training=True):
    """
    define the model, we use slim's implemention of resnet
    """
    images = mean_image_subtraction(images)

    with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=weight_decay)):
        logits, end_points = resnet_v1.resnet_v1_50(
            images, is_training=is_training, scope='resnet_v1_50')

    with tf.variable_scope('feature_fusion', values=[end_points.values]):
        batch_norm_params = {
            'decay': 0.997,
            'epsilon': 1e-5,
            'scale': True,
            'is_training': is_training
        }
        with slim.arg_scope([slim.conv2d],
                            activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(weight_decay)):
            f = [end_points['pool5'], end_points['pool4'],
                 end_points['pool3'], end_points['pool2']]
            for i in range(4):
                print('Shape of f_{} {}'.format(i, f[i].shape))
            g = [None, None, None, None]
            h = [None, None, None, None]
            num_outputs = [None, 128, 64, 32]
            for i in range(4):
                if i == 0:
                    h[i] = f[i]
                else:
                    c1_1 = slim.conv2d(
                        tf.concat([g[i - 1], f[i]], axis=-1), num_outputs[i], 1)
                    h[i] = slim.conv2d(c1_1, num_outputs[i], 3)
                if i <= 2:
                    g[i] = unpool(h[i])
                else:
                    g[i] = slim.conv2d(h[i], num_outputs[i], 3)
                print('Shape of h_{} {}, g_{} {}'.format(
                    i, h[i].shape, i, g[i].shape))

            # here we use a slightly different way for regression part,
            # we first use a sigmoid to limit the regression range, and also
            # this is do with the angle map
            F_score = slim.conv2d(
                g[3], 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None)
            # 4 channel of axis aligned bbox and 1 channel rotation angle
            geo_map = slim.conv2d(
                g[3], 4, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) * text_scale
            angle_map = (slim.conv2d(g[3], 1, 1, activation_fn=tf.nn.sigmoid,
                                     normalizer_fn=None) - 0.5) * np.pi / 2  # angle is between [-45, 45]
            F_geometry = tf.concat([geo_map, angle_map], axis=-1)

    return F_score, F_geometry


def dice_coefficient(y_true_cls, y_pred_cls, training_mask):
    """
    dice loss
    :param y_true_cls:
    :param y_pred_cls:
    :param training_mask:
    :return:
    """
    eps = 1e-5
    intersection = tf.reduce_sum(y_true_cls * y_pred_cls * training_mask)
    union = tf.reduce_sum(y_true_cls * training_mask) + \
            tf.reduce_sum(y_pred_cls * training_mask) + eps
    loss = 1. - (2 * intersection / union)
    tf.summary.scalar('classification_dice_loss', loss)
    return loss


def get_loss(y_true_cls, y_pred_cls,
             y_true_geo, y_pred_geo,
             training_mask):
    """
    define the loss used for training, contraning two part,
    the first part we use dice loss instead of weighted logloss,
    the second part is the iou loss defined in the paper
    :param y_true_cls: ground truth of text
    :param y_pred_cls: prediction os text
    :param y_true_geo: ground truth of geometry
    :param y_pred_geo: prediction of geometry
    :param training_mask: mask used in training, to ignore some text annotated by ###
    :return:
    """

    """
    Section: EAST : 3.4.2 Loss for Geometries
      p0 : d1                  p1 : d2
       --------------------------
      |                          |
      |                          |
      |                          |
       --------------------------  
     p3 : d4                   p2 : d3

     where d1,d2,d3 and d4 represents the distance from a pixel to the top, right, bottom and 
     left boundary of its corresponding rectangle, respectively. 
    """

    classification_loss = dice_coefficient(y_true_cls, y_pred_cls, training_mask)

    # scale classification loss to match the iou loss part
    classification_loss *= 0.01

    # p0 -> top, p1->right, p2->bottom, p3->left
    p0_gt, p1_gt, p2_gt, p3_gt, theta_gt = tf.split(
        value=y_true_geo, num_or_size_splits=5, axis=3)
    p0_pred, p1_pred, p2_pred, p3_pred, theta_pred = tf.split(
        value=y_pred_geo, num_or_size_splits=5, axis=3)

    area_gt = (p0_gt + p2_gt) * (p1_gt + p3_gt)
    area_pred = (p0_pred + p2_pred) * (p1_pred + p3_pred)

    w_union = tf.minimum(p1_gt, p1_pred) + tf.minimum(p3_gt, p3_pred)
    h_union = tf.minimum(p0_gt, p0_pred) + tf.minimum(p2_gt, p2_pred)
    area_intersect = w_union * h_union

    area_union = area_gt + area_pred - area_intersect

    L_AABB = -tf.log((area_intersect + 1.0) / (area_union + 1.0))
    L_theta = 1 - tf.cos(theta_pred - theta_gt)
    L_g = L_AABB + 20 * L_theta

    tf.summary.scalar('geometry_AABB', tf.reduce_mean(
        L_AABB * y_true_cls * training_mask))
    tf.summary.scalar('geometry_theta', tf.reduce_mean(
        L_theta * y_true_cls * training_mask))

    return tf.reduce_mean(L_g * y_true_cls * training_mask) + classification_loss


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)

        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)

    return average_grads

def decode(serialized_example):
    print_info("Decode")
    # 1. define a parser
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            _transformed_name('images'): tf.FixedLenFeature([512 * 512 * 3], tf.float32),
            _transformed_name('score_maps'): tf.FixedLenFeature([128 * 128 * 1], tf.float32),
            _transformed_name('geo_maps'): tf.FixedLenFeature([128 * 128 * 5], tf.float32),
            _transformed_name('training_masks'): tf.FixedLenFeature([128 * 128 * 1], tf.float32),
        })

    image = tf.reshape(
        tf.cast(features[_transformed_name('images')], tf.float32), shape=[512, 512, 3])
    score_map = tf.reshape(
        tf.cast(features[_transformed_name('score_maps')], tf.float32), shape=[128, 128, 1])
    geo_map = tf.reshape(
        tf.cast(features[_transformed_name('geo_maps')], tf.float32), shape=[128, 128, 5])
    training_masks = tf.reshape(
        tf.cast(features[_transformed_name('training_masks')], tf.float32), shape=[128, 128, 1])

    return {_transformed_name("images"): image, _transformed_name("score_maps"): score_map,
            _transformed_name("geo_maps"): geo_map, _transformed_name("training_masks"): training_masks}, training_masks

#TODO move this
class EASTModel:
    def __init__(self,
                 learning_rate=0.0001,
                 model_root_directory="./",
                 moving_average_decay=0.997):
        print_info("EASTIEstimatorModel")
        print_info("model_root_directory : {}".format(model_root_directory))
        print_info("learning_rate : {}".format(learning_rate))

        self._model_root_directory = model_root_directory
        self._learning_rate = learning_rate
        self._moving_average_decay = moving_average_decay

    def _get_optimizer(self, loss):
        tower_grads = []
        with tf.name_scope("optimizer") as scope:
            global_step = tf.train.get_global_step()
            learning_rate = tf.train.exponential_decay(self._learning_rate,
                                                       global_step,
                                                       decay_steps=100,
                                                       decay_rate=0.94,
                                                       staircase=True)
            # add summary
            tf.summary.scalar('learning_rate', learning_rate)

            optimizer = tf.train.AdamOptimizer(
                learning_rate=learning_rate,
                beta1=0.9,
                beta2=0.999,
                epsilon=1e-8)

            batch_norm_updates_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS))  # TODO scope
            grads = optimizer.compute_gradients(loss)
            tower_grads.append(grads)
            grads = average_gradients(tower_grads)
            apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)

            variable_averages = tf.train.ExponentialMovingAverage(self._moving_average_decay,
                                                                  global_step)
            variables_averages_op = variable_averages.apply(tf.trainable_variables())

            # batch norm updates
            with tf.control_dependencies([variables_averages_op, apply_gradient_op, batch_norm_updates_op]):
                train_op = tf.no_op(name='train_op')

        return train_op

    def __call__(self, features, labels, params, mode, config=None):
        """
        Used for the :tf_main:`model_fn <estimator/Estimator#__init__>`
        argument when constructing
        :tf_main:`tf.estimator.Estimator <estimator/Estimator>`.
        """
        return self._build(features, labels, params, mode, config=config)

    @property
    def model_dir(self):
        """
        Returns model directory `model_root_directory`/`experiment_name`/VanillaGAN
        :return:
        """
        return os.path.join(self._model_root_directory,
                            type(self).__name__)

    def _build(self, features, labels, params, mode, config=None):
        # features = decode(features)

        input_images = features[_transformed_name('images')]
        tf.logging.info(":=>")
        tf.logging.info(input_images.get_shape())

        is_training = mode == tf.estimator.ModeKeys.TRAIN

        # Build inference graph
        with tf.variable_scope(tf.get_variable_scope(), reuse=False):
            f_score, f_geometry = model(input_images, is_training=is_training)

        loss = None
        optimizer = None
        predictions = {"f_score": f_score, "f_geometry": f_geometry}

        if mode != tf.estimator.ModeKeys.PREDICT:
            input_score_maps = features[_transformed_name('score_maps')]
            input_geo_maps = features[_transformed_name('geo_maps')]
            input_training_masks = features[_transformed_name('training_masks')]

            model_loss = get_loss(input_score_maps,
                                  f_score,
                                  input_geo_maps,
                                  f_geometry,
                                  input_training_masks)
            loss = tf.add_n(
                [model_loss] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

            # add summary
            # if reuse_variables is None:
            tf.summary.image('input', input_images)
            tf.summary.image('score_map', input_score_maps)
            tf.summary.image('score_map_pred', f_score * 255)
            tf.summary.image('geo_map_0', input_geo_maps[:, :, :, 0:1])
            tf.summary.image('geo_map_0_pred', f_geometry[:, :, :, 0:1])
            tf.summary.image('training_masks', input_training_masks)
            tf.summary.scalar('model_loss', model_loss)
            tf.summary.scalar('total_loss', loss)

            optimizer = self._get_optimizer(loss=loss)

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs={'predict': tf.estimator.export.PredictOutput(predictions)},
            loss=loss,
            train_op=optimizer,
            eval_metric_ops=None)
