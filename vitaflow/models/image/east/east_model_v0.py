import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
import vitaflow.models.image.east.utils.resnet_v1 as resnet_v1
import gin

from vitaflow.utils.print_helper import print_info
from vitaflow.utils.registry import register_model
from vitaflow.models.interface_model import IEstimatorModel


def unpool(inputs):
    return tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1]*2,  tf.shape(inputs)[2]*2])


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
    define the _model, we use slim's implemention of resnet
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
                        tf.concat([g[i-1], f[i]], axis=-1), num_outputs[i], 1)
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
                                     normalizer_fn=None) - 0.5) * np.pi/2  # angle is between [-45, 45]
            F_geometry = tf.concat([geo_map, angle_map], axis=-1)

    return F_score, F_geometry


def dice_coefficient(y_true_cls, y_pred_cls, training_mask=None):
    """
    dice loss
    :param y_true_cls:
    :param y_pred_cls:
    :param training_mask:
    :return:
    """
    eps = 1e-5
    # intersection = tf.reduce_sum(y_true_cls * y_pred_cls * training_mask)
    intersection = tf.reduce_sum(y_true_cls * y_pred_cls)
    #
    # union = tf.reduce_sum(y_true_cls * training_mask) + \
    #         tf.reduce_sum(y_pred_cls * training_mask) + eps


    union = tf.reduce_sum(y_true_cls) + \
            tf.reduce_sum(y_pred_cls) + eps

    loss = 1. - (2 * intersection / union)
    tf.summary.scalar('classification_dice_loss', loss)
    return loss


def get_loss(y_true_cls, y_pred_cls,
             y_true_geo, y_pred_geo,
             training_mask=None):
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
      p3 : d4                  p2 : d3
       --------------------------
      |                          |
      |                          |
      |                          |
       --------------------------  
     p0 : d1                  p1 : d2
     
     where d1,d2,d3 and d4 represents the distance from a pixel to the top, right, bottom and 
     left boundary of its corresponding rectangle, respectively. 
    """

    # classification_loss = dice_coefficient(y_true_cls, y_pred_cls, training_mask)
    classification_loss = dice_coefficient(y_true_cls, y_pred_cls)

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

    L_AABB = -tf.log((area_intersect + 1.0)/(area_union + 1.0))
    L_theta = 1 - tf.cos(theta_pred - theta_gt)
    L_g = L_AABB + 20 * L_theta

    # tf.summary.scalar('geometry_AABB', tf.reduce_mean(
    #     L_AABB * y_true_cls * training_mask))
    # tf.summary.scalar('geometry_theta', tf.reduce_mean(
    #     L_theta * y_true_cls * training_mask))

    tf.summary.scalar('geometry_AABB', tf.reduce_mean(
        L_AABB * y_true_cls))
    tf.summary.scalar('geometry_theta', tf.reduce_mean(
        L_theta * y_true_cls))

    # return tf.reduce_mean(L_g * y_true_cls * training_mask) + classification_loss
    return tf.reduce_mean(L_g * y_true_cls) + classification_loss



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

# -----------------------------------------------------------------------------------------------------------

@register_model
@gin.configurable
class EASTSlimModel(IEstimatorModel):
    def __init__(self,
                 experiment_name,
                 dataset,
                 learning_rate=0.0001,
                 model_root_directory=gin.REQUIRED,
                 moving_average_decay=0.997):
        IEstimatorModel.__init__(self,
                                 experiment_name=experiment_name,
                                 model_root_directory=model_root_directory,
                                 dataset=dataset)
        version = (tf.version.VERSION).split(".")

        if int(version[0]) > 1 or int(version[1]) > 13:
            print_info("Selected _model is doesn't supports current Tensorflow version. Use Tensorflow 1.13.0 or before!")
            exit(0)

        self._model_root_directory = model_root_directory
        self._learning_rate = learning_rate
        self._moving_average_decay = moving_average_decay

    def _get_optimizer(self, loss):
        tower_grads = []
        with tf.name_scope("optimizer") as scope:

            global_step=tf.train.get_global_step()
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

            batch_norm_updates_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS)) #TODO scope
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

    @property
    def model_dir(self):
        """
        Returns _model directory `model_root_directory`/`experiment_name`/VanillaGAN
        :return:
        """
        return os.path.join(self._model_root_directory,
                            type(self).__name__)

    def _build(self, features, labels, params, mode, config=None):

        input_images = features['images']

        is_training = mode == tf.estimator.ModeKeys.TRAIN

        # Build inference graph
        with tf.variable_scope(tf.get_variable_scope(), reuse=False):
            f_score, f_geometry = model(input_images, is_training=is_training)

        loss = None
        optimizer = None
        predictions = {"f_score" : f_score, "f_geometry" : f_geometry}

        if mode != tf.estimator.ModeKeys.PREDICT:
            input_score_maps = features['score_maps']
            input_geo_maps = features['geo_maps']
            # input_training_masks = features['training_masks']

            model_loss = get_loss(input_score_maps,
                                  f_score,
                                  input_geo_maps,
                                  f_geometry)#,
                                  # input_training_masks)
            loss = tf.add_n(
                [model_loss] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

            # add summary
            # if reuse_variables is None:
            tf.summary.image('input', input_images)
            tf.summary.image('score_map', input_score_maps)
            tf.summary.image('score_map_pred', f_score * 255)
            tf.summary.image('geo_map_0', input_geo_maps[:, :, :, 0:1])
            tf.summary.image('geo_map_0_pred', f_geometry[:, :, :, 0:1])
            # tf.summary.image('training_masks', input_training_masks)
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