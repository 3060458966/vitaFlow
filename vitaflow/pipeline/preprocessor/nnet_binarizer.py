import numpy as np
from keras.optimizers import Adam

# from model.unet import unet
from vitaflow.pipeline.preprocessor.img_processing import *

import os
import subprocess

from vitaflow.pipeline.interfaces.plugin import ImagePluginInterface
from vitaflow import demo_config

import shlex
import argparse
import glob
import time
from tqdm import tqdm

from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.layers.core import SpatialDropout2D, Activation
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.models import Model


def double_conv_layer(inputs, filter):
    conv = Conv2D(filter, (3, 3), padding='same', kernel_initializer='he_normal')(inputs)
    conv = BatchNormalization(axis=3)(conv)
    conv = Activation('relu')(conv)
    conv = Conv2D(filter, (3, 3), padding='same', kernel_initializer='he_normal')(conv)
    conv = BatchNormalization(axis=3)(conv)
    conv = Activation('relu')(conv)
    conv = SpatialDropout2D(0.1)(conv)
    return conv


def down_layer(inputs, filter):
    """Create downsampling layer."""
    conv = double_conv_layer(inputs, filter)
    pool = MaxPooling2D(pool_size=(2, 2))(conv)
    return conv, pool


def up_layer(inputs, concats, filter):
    """Create upsampling layer."""
    return double_conv_layer(concatenate([UpSampling2D(size=(2, 2))(inputs), concats], axis=3), filter)


def unet():
    """Create U-net."""
    inputs = Input((128, 128, 1))

    # Downsampling.
    down1, pool1 = down_layer(inputs, 32)
    down2, pool2 = down_layer(pool1, 64)
    down3, pool3 = down_layer(pool2, 128)
    down4, pool4 = down_layer(pool3, 256)
    down5, pool5 = down_layer(pool4, 512)

    # Bottleneck.
    bottleneck = double_conv_layer(pool5, 1024)

    # Upsampling.
    up5 = up_layer(bottleneck, down5, 512)
    up4 = up_layer(up5, down4, 256)
    up3 = up_layer(up4, down3, 128)
    up2 = up_layer(up3, down2, 64)
    up1 = up_layer(up2, down1, 32)

    outputs = Conv2D(1, (1, 1))(up1)
    outputs = Activation('sigmoid')(outputs)

    model = Model(inputs, outputs)

    return model


def nnet_binarizes(in_file_path, out_file_path, model, batch_size=10):
    # start_time = time.time()
    # fnames_in = list(glob.iglob(os.path.join(in_file_path, '**', '*.jpg*'), recursive=True))
    # for fname in tqdm(fnames_in):
    img = cv2.imread(in_file_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    img = binarize_img(img, model, batch_size)
    # cv2.imwrite(os.path.join(out_file_path, os.path.split(in_file_path)[-1]), img)
    cv2.imwrite(out_file_path, img)


    # print("finished in {0:.2f} seconds".format(time.time() - start_time))

