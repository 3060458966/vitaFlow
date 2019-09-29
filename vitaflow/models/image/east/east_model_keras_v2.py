# Reference: https://github.com/kurapan/EAST
import io
import numpy as np
import six
import copy
from six.moves import zip
import gin
import tensorflow as tf
from PIL import Image

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, concatenate, BatchNormalization, Lambda, Input, multiply, add, ZeroPadding2D, Activation, Layer, MaxPooling2D, Dropout
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint, Callback

import tensorflow.keras.backend as K

from keras.utils.generic_utils import serialize_keras_object
from keras.utils.generic_utils import deserialize_keras_object

# import keras
# from keras.applications.resnet50 import ResNet50
from keras.models import Model
# from keras.layers import Conv2D, concatenate, BatchNormalization, Lambda, Input, multiply, add, ZeroPadding2D, Activation, Layer, MaxPooling2D, Dropout
# from keras import regularizers
# import tensorflow as tf
# from keras.optimizers import Optimizer
#

from keras.legacy import interfaces
# from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint, Callback


from vitaflow.models.interface_model import IKerasModel
from vitaflow.utils.print_helper import print_debug
from vitaflow.utils.registry import register_model

RESIZE_FACTOR = 2


def resize_bilinear(x):
    return tf.image.resize(x, size=[K.shape(x)[1]*RESIZE_FACTOR, K.shape(x)[2]*RESIZE_FACTOR])


def resize_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 4
    shape[1] *= RESIZE_FACTOR
    shape[2] *= RESIZE_FACTOR
    return tuple(shape)


def dice_loss(overly_small_text_region_training_mask, text_region_boundary_training_mask, loss_weight, small_text_weight):
    def loss(y_true, y_pred):
        eps = 1e-5
        _training_mask = tf.minimum(overly_small_text_region_training_mask + small_text_weight, 1) * text_region_boundary_training_mask
        intersection = tf.reduce_sum(y_true * y_pred * _training_mask)
        union = tf.reduce_sum(y_true * _training_mask) + tf.reduce_sum(y_pred * _training_mask) + eps
        loss = 1. - (2. * intersection / union)
        return loss * loss_weight
    return loss


def rbox_loss(overly_small_text_region_training_mask, text_region_boundary_training_mask, small_text_weight, target_score_map):
    def loss(y_true, y_pred):
        # d1 -> top, d2->right, d3->bottom, d4->left
        d1_gt, d2_gt, d3_gt, d4_gt, theta_gt = tf.split(value=y_true, num_or_size_splits=5, axis=3)
        d1_pred, d2_pred, d3_pred, d4_pred, theta_pred = tf.split(value=y_pred, num_or_size_splits=5, axis=3)
        area_gt = (d1_gt + d3_gt) * (d2_gt + d4_gt)
        area_pred = (d1_pred + d3_pred) * (d2_pred + d4_pred)
        w_union = tf.minimum(d2_gt, d2_pred) + tf.minimum(d4_gt, d4_pred)
        h_union = tf.minimum(d1_gt, d1_pred) + tf.minimum(d3_gt, d3_pred)
        area_intersect = w_union * h_union
        area_union = area_gt + area_pred - area_intersect
        L_AABB = -tf.math.log((area_intersect + 1.0)/(area_union + 1.0))
        L_theta = 1 - tf.math.cos(theta_pred - theta_gt)
        L_g = L_AABB + 20 * L_theta
        _training_mask = tf.minimum(overly_small_text_region_training_mask + small_text_weight, 1) * text_region_boundary_training_mask
        return tf.reduce_mean(L_g * target_score_map * _training_mask)
    return loss


class AdamW(Optimizer):
    """Adam optimizer.
    Default parameters follow those provided in the original paper.
    # Arguments
        lr: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor.
        decay: float >= 0. Learning rate decay over each update.
        weight_decay: float >= 0. Decoupled weight decay over each update.
    # References
        - [Adam - A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980v8)
        - [Optimization for Deep Learning Highlights in 2017](http://ruder.io/deep-learning-optimization-2017/index.html)
        - [Fixing Weight Decay Regularization in Adam](https://arxiv.org/abs/1711.05101)
    """

    def __init__(self,
                 name="AdamW",
                 lr=0.001,
                 beta_1=0.9,
                 beta_2=0.999,
                 weight_decay=1e-4,  # decoupled weight decay (1/4)
                 epsilon=1e-8,
                 decay=0.,
                 **kwargs):
        try:
            super(AdamW, self).__init__(name=name, **kwargs)
        except:
            super(AdamW, self).__init__(**kwargs)

        from pprint import pprint
        pprint(vars(self))
        with K.name_scope(self.__class__.__name__):
            self.lr = tf.Variable(lr, name='AdamW_lr')
            self.iterations = tf.Variable(0, dtype='int64', name='iterations')
            self.beta_1 = tf.Variable(beta_1, name='beta_1')
            self.beta_2 = tf.Variable(beta_2, name='beta_2')
            self.decay = tf.Variable(decay, name='decay')
            self.wd = tf.Variable(weight_decay, name='weight_decay') # decoupled weight decay (2/4)
        self.epsilon = epsilon
        self.initial_decay = decay

    def __getattribute__(self, name):
        """Overridden to support hyperparameter access."""
        try:
            if name == "lr":
                name = "learning_rate"
            return super(Optimizer, self).__getattribute__(name)
        except AttributeError as e:
            # Needed to avoid infinite recursion with __setattr__.
            if name == "_hyper":
                raise e
            # Backwards compatibility with Keras optimizers.
            if name == "lr":
                name = "learning_rate"
            if name in self._hyper:
                return self._get_hyper(name)
            raise e

    # @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]
        wd = self.wd # decoupled weight decay (3/4)

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * K.cast(self.iterations,
                                                  K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1
        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                     (1. - K.pow(self.beta_1, t)))

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        self.weights = [self.iterations] + ms + vs

        for p, g, m, v in zip(params, grads, ms, vs):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon) - lr * wd * p # decoupled weight decay (4/4)

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'decay': float(K.get_value(self.decay)),
                  'weight_decay': float(K.get_value(self.wd)),
                  'epsilon': self.epsilon}
        base_config = super(AdamW, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def make_image_summary(tensor):
    """
    Convert an numpy representation image to Image protobuf.
    Copied from https://github.com/lanpa/tensorboard-pytorch/
    """
    if len(tensor.shape) == 2:
        height, width = tensor.shape
        channel = 1
    else:
        height, width, channel = tensor.shape
        if channel == 1:
            tensor = tensor[:, :, 0]
    image = Image.fromarray(tensor)
    output = io.BytesIO()
    image.save(output, format='PNG')
    image_string = output.getvalue()
    output.close()
    return tf.Summary.Image(height=height,
                            width=width,
                            colorspace=channel,
                            encoded_image_string=image_string)

class CustomTensorBoard(TensorBoard):
    def __init__(self,
                 input_size,
                 log_dir,
                 score_map_loss_weight,
                 small_text_weight,
                 data_generator,
                 write_graph=False):
        self.score_map_loss_weight = score_map_loss_weight
        self.small_text_weight = small_text_weight
        self.data_generator = data_generator
        self.input_size = input_size
        super(CustomTensorBoard, self).__init__(log_dir=log_dir, write_graph=write_graph)

    def on_epoch_end(self, epoch, logs=None):
        logs.update({'learning_rate': K.eval(self.model.optimizer.lr), 'small_text_weight': K.eval(self.small_text_weight)})
        data = next(self.data_generator)
        pred_score_maps, pred_geo_maps = self.model.predict([data[0][0], data[0][1], data[0][2], data[0][3]])
        img_summaries = []
        for i in range(3):
            input_image_summary = make_image_summary(((data[0][0][i] + 1) * 127.5).astype('uint8'))
            overly_small_text_region_training_mask_summary = make_image_summary((data[0][1][i] * 255).astype('uint8'))
            text_region_boundary_training_mask_summary = make_image_summary((data[0][2][i] * 255).astype('uint8'))
            target_score_map_summary = make_image_summary((data[1][0][i] * 255).astype('uint8'))
            pred_score_map_summary = make_image_summary((pred_score_maps[i] * 255).astype('uint8'))
            img_summaries.append(tf.Summary.Value(tag='input_image/%d' % i, image=input_image_summary))
            img_summaries.append(tf.Summary.Value(tag='overly_small_text_region_training_mask/%d' % i, image=overly_small_text_region_training_mask_summary))
            img_summaries.append(tf.Summary.Value(tag='text_region_boundary_training_mask/%d' % i, image=text_region_boundary_training_mask_summary))
            img_summaries.append(tf.Summary.Value(tag='score_map_target/%d' % i, image=target_score_map_summary))
            img_summaries.append(tf.Summary.Value(tag='score_map_pred/%d' % i, image=pred_score_map_summary))
            for j in range(4):
                target_geo_map_summary = make_image_summary((data[1][1][i, :, :, j] / self.input_size * 255).astype('uint8'))
                pred_geo_map_summary = make_image_summary((pred_geo_maps[i, :, :, j] / self.input_size * 255).astype('uint8'))
                img_summaries.append(tf.Summary.Value(tag='geo_map_%d_target/%d' % (j, i), image=target_geo_map_summary))
                img_summaries.append(tf.Summary.Value(tag='geo_map_%d_pred/%d' % (j, i), image=pred_geo_map_summary))
            target_geo_map_summary = make_image_summary(((data[1][1][i, :, :, 4] + 1) * 127.5).astype('uint8'))
            pred_geo_map_summary = make_image_summary(((pred_geo_maps[i, :, :, 4] + 1) * 127.5).astype('uint8'))
            img_summaries.append(tf.Summary.Value(tag='geo_map_%d_target/%d' % (4, i), image=target_geo_map_summary))
            img_summaries.append(tf.Summary.Value(tag='geo_map_%d_pred/%d' % (4, i), image=pred_geo_map_summary))
        tf_summary = tf.Summary(value=img_summaries)
        self.writer.add_summary(tf_summary, epoch + 1)
        super(CustomTensorBoard, self).on_epoch_end(epoch + 1, logs)


class ValidationEvaluator(Callback):
    def __init__(self, validation_data, validation_log_dir, batch_size, period=5):
        super(Callback, self).__init__()

        self.period = period
        self.validation_data = validation_data
        self.validation_log_dir = validation_log_dir
        self.val_writer = tf.summary.create_file_writer(self.validation_log_dir)  # tf.summary.FileWriter
        self.batch_size = batch_size

    def on_epoch_end(self, epoch, logs={}):
        if (epoch + 1) % self.period == 0:
            val_loss, val_score_map_loss, val_geo_map_loss = self.model.evaluate([self.validation_data[0],
                                                                                  self.validation_data[1],
                                                                                  self.validation_data[2],
                                                                                  self.validation_data[3]],
                                                                                 [self.validation_data[3],
                                                                                  self.validation_data[4]],
                                                                                 batch_size=self.batch_size)
            print('\nEpoch %d: val_loss: %.4f, val_score_map_loss: %.4f, val_geo_map_loss: %.4f' % (epoch + 1, val_loss, val_score_map_loss, val_geo_map_loss))
            val_loss_summary = tf.Summary()
            val_loss_summary_value = val_loss_summary.value.add()
            val_loss_summary_value.simple_value = val_loss
            val_loss_summary_value.tag = 'loss'
            self.val_writer.add_summary(val_loss_summary, epoch + 1)
            val_score_map_loss_summary = tf.Summary()
            val_score_map_loss_summary_value = val_score_map_loss_summary.value.add()
            val_score_map_loss_summary_value.simple_value = val_score_map_loss
            val_score_map_loss_summary_value.tag = 'pred_score_map_loss'
            self.val_writer.add_summary(val_score_map_loss_summary, epoch + 1)
            val_geo_map_loss_summary = tf.Summary()
            val_geo_map_loss_summary_value = val_geo_map_loss_summary.value.add()
            val_geo_map_loss_summary_value.simple_value = val_geo_map_loss
            val_geo_map_loss_summary_value.tag = 'pred_geo_map_loss'
            self.val_writer.add_summary(val_geo_map_loss_summary, epoch + 1)

            pred_score_maps, pred_geo_maps = self.model.predict([self.validation_data[0][0:3], self.validation_data[1][0:3],
                                                                 self.validation_data[2][0:3], self.validation_data[3][0:3]])
            img_summaries = []
            for i in range(3):
                input_image_summary = make_image_summary(((self.validation_data[0][i] + 1) * 127.5).astype('uint8'))
                overly_small_text_region_training_mask_summary = make_image_summary((self.validation_data[1][i] * 255).astype('uint8'))
                text_region_boundary_training_mask_summary = make_image_summary((self.validation_data[2][i] * 255).astype('uint8'))
                target_score_map_summary = make_image_summary((self.validation_data[3][i] * 255).astype('uint8'))
                pred_score_map_summary = make_image_summary((pred_score_maps[i] * 255).astype('uint8'))
                img_summaries.append(tf.Summary.Value(tag='input_image/%d' % i, image=input_image_summary))
                img_summaries.append(tf.Summary.Value(tag='overly_small_text_region_training_mask/%d' % i, image=overly_small_text_region_training_mask_summary))
                img_summaries.append(tf.Summary.Value(tag='text_region_boundary_training_mask/%d' % i, image=text_region_boundary_training_mask_summary))
                img_summaries.append(tf.Summary.Value(tag='score_map_target/%d' % i, image=target_score_map_summary))
                img_summaries.append(tf.Summary.Value(tag='score_map_pred/%d' % i, image=pred_score_map_summary))
                for j in range(4):
                    target_geo_map_summary = make_image_summary((self.validation_data[4][i, :, :, j] / self.input_size * 255).astype('uint8'))
                    pred_geo_map_summary = make_image_summary((pred_geo_maps[i, :, :, j] / self.input_size * 255).astype('uint8'))
                    img_summaries.append(tf.Summary.Value(tag='geo_map_%d_target/%d' % (j, i), image=target_geo_map_summary))
                    img_summaries.append(tf.Summary.Value(tag='geo_map_%d_pred/%d' % (j, i), image=pred_geo_map_summary))
                target_geo_map_summary = make_image_summary(((self.validation_data[4][i, :, :, 4] + 1) * 127.5).astype('uint8'))
                pred_geo_map_summary = make_image_summary(((pred_geo_maps[i, :, :, 4] + 1) * 127.5).astype('uint8'))
                img_summaries.append(tf.Summary.Value(tag='geo_map_%d_target/%d' % (4, i), image=target_geo_map_summary))
                img_summaries.append(tf.Summary.Value(tag='geo_map_%d_pred/%d' % (4, i), image=pred_geo_map_summary))
            tf_summary = tf.Summary(value=img_summaries)
            self.val_writer.add_summary(tf_summary, epoch + 1)
            self.val_writer.flush()


class SmallTextWeight(Callback):
    def __init__(self, weight):
        self.weight = weight

    # TO BE CHANGED
    def on_epoch_end(self, epoch, logs={}):
        #K.set_value(self.weight, np.minimum(epoch / (0.5 * FLAGS.max_epochs), 1.))
        K.set_value(self.weight, 0)


@register_model
@gin.configurable
class EASTV2Keras(IKerasModel):

    def __init__(self,
                 experiment_name=gin.REQUIRED,
                 model_root_directory=gin.REQUIRED,
                 dataset=None,
                 init_learning_rate=0.0001,
                 input_size=512):
        IKerasModel.__init__(self,
                             experiment_name=experiment_name,
                             model_root_directory=model_root_directory,
                             dataset=dataset)

        self._init_learning_rate = init_learning_rate
        input_image = Input(shape=(None, None, 3), name='input_image')
        overly_small_text_region_training_mask = Input(shape=(None, None, 1), name='overly_small_text_region_training_mask')
        text_region_boundary_training_mask = Input(shape=(None, None, 1), name='text_region_boundary_training_mask')
        target_score_map = Input(shape=(None, None, 1), name='target_score_map')
        resnet = ResNet50(input_tensor=input_image, weights='imagenet', include_top=False, pooling=None)
        print_debug(resnet.summary())
        x = resnet.get_layer('activation_49').output  # activation_49 -> conv5_block3_out

        x = Lambda(resize_bilinear, name='resize_1')(x)
        x = concatenate([x, resnet.get_layer('activation_40').output], axis=3)  # activation_40 -> concat_conv4_block6_out
        x = Conv2D(128, (1, 1), padding='same', kernel_regularizer=regularizers.l2(1e-5))(x)
        x = BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(x)
        x = Activation('relu')(x)
        x = Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(1e-5))(x)
        x = BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(x)
        x = Activation('relu')(x)

        x = Lambda(resize_bilinear, name='resize_2')(x)
        x = concatenate([x, resnet.get_layer('activation_22').output], axis=3) # activation_22 -> conv3_block4_out
        x = Conv2D(64, (1, 1), padding='same', kernel_regularizer=regularizers.l2(1e-5))(x)
        x = BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(x)
        x = Activation('relu')(x)
        x = Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(1e-5))(x)
        x = BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(x)
        x = Activation('relu')(x)

        x = Lambda(resize_bilinear, name='resize_3')(x)
        x = concatenate([x, ZeroPadding2D(((1, 0),(1, 0)))(resnet.get_layer('activation_10').output)], axis=3) # activation_10 -> conv2_block3_out
        x = Conv2D(32, (1, 1), padding='same', kernel_regularizer=regularizers.l2(1e-5))(x)
        x = BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(x)
        x = Activation('relu')(x)
        x = Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(1e-5))(x)
        x = BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(x)
        x = Activation('relu')(x)

        x = Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(1e-5))(x)
        x = BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(x)
        x = Activation('relu')(x)

        pred_score_map = Conv2D(1, (1, 1), activation=tf.nn.sigmoid, name='pred_score_map')(x)
        rbox_geo_map = Conv2D(4, (1, 1), activation=tf.nn.sigmoid, name='rbox_geo_map')(x)
        rbox_geo_map = Lambda(lambda x: x * input_size)(rbox_geo_map)
        angle_map = Conv2D(1, (1, 1), activation=tf.nn.sigmoid, name='rbox_angle_map')(x)
        angle_map = Lambda(lambda x: (x - 0.5) * np.pi / 2)(angle_map)
        pred_geo_map = concatenate([rbox_geo_map, angle_map], axis=3, name='pred_geo_map')

        self.input_image = input_image
        self.overly_small_text_region_training_mask = overly_small_text_region_training_mask
        self.text_region_boundary_training_mask = text_region_boundary_training_mask
        self.target_score_map = target_score_map
        self.pred_score_map = pred_score_map
        self.pred_geo_map = pred_geo_map

        self.score_map_loss_weight = tf.Variable(0.01, name='score_map_loss_weight')

        self.small_text_weight = tf.Variable(0., name='small_text_weight')

        self.small_text_weight_callback = SmallTextWeight(self.small_text_weight)

        train_data_generator = dataset.get_train_dataset_gen()
        self.tb = CustomTensorBoard(log_dir=model_root_directory + '/train',
                                    score_map_loss_weight=self.score_map_loss_weight,
                                    small_text_weight=self.small_text_weight,
                                    data_generator=train_data_generator,
                                    write_graph=True,
                                    input_size=512)
        validation_evaluator = ValidationEvaluator(dataset.get_val_dataset_gen(),
                                                   validation_log_dir=model_root_directory + '/val',
                                                   batch_size=dataset.batch_size)

        self._callbacks = [self.tb, self.small_text_weight_callback, validation_evaluator]

    def get_inputs(self):
        return [self.input_image, self.overly_small_text_region_training_mask, self.text_region_boundary_training_mask, self.target_score_map]

    def get_outputs(self):
        return [self.pred_score_map, self.pred_geo_map]

    def get_loss(self, labels=None, logits=None):
        loss = [dice_loss(self.overly_small_text_region_training_mask, self.text_region_boundary_training_mask,
                          self.score_map_loss_weight, self.small_text_weight),
                rbox_loss(self.overly_small_text_region_training_mask, self.text_region_boundary_training_mask,
                          self.small_text_weight, self.target_score_map)]

        return loss

    def get_optimizer(self, loss=None):
        return "adam" #AdamW(lr=self._init_learning_rate)

    def get_callbacks(self):
        return self._callbacks