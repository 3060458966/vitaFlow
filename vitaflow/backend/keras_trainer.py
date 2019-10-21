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

import io

from vitaflow.backend.interface_trainer import TrainerBase


import os
import sys
import time
from tensorflow.keras.models import Model
from vitaflow.backend.interface_trainer import TrainerBase
from vitaflow.datasets.interface_dataset import IDataset
from vitaflow.models.interface_model import IKerasModel
from vitaflow.utils.print_helper import print_info

from tensorflow.keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint, Callback
import numpy as np
import tensorflow as tf
from PIL import Image

from keras.models import Model

def lr_decay(init_learning_rate, lr_decay_rate, lr_decay_steps, epoch):
    return init_learning_rate * np.power(lr_decay_rate, epoch // lr_decay_steps)


class CustomModelCheckpoint(Callback):
    def __init__(self, model, path, period, save_weights_only):
        super(CustomModelCheckpoint, self).__init__()
        self.period = period
        self.path = path
        # We set the _model (non multi gpu) under an other name
        self.model_for_saving = model
        self.epochs_since_last_save = 0
        self.save_weights_only = save_weights_only

    def on_epoch_end(self, epoch, logs=None):
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            if self.save_weights_only:
                self.model_for_saving.save_weights(self.path.format(epoch=epoch + 1, **logs), overwrite=True)
            else:
                self.model_for_saving.save(self.path.format(epoch=epoch + 1, **logs), overwrite=True)


class KerasTrainer(TrainerBase):
    def __init__(self,
                 model: IKerasModel,
                 dataset: IDataset,
                 nb_workers=4,
                 model_store_path=""):
        TrainerBase.__init__(self,
                             model_store_path=model_store_path,
                             dataset=dataset,
                             model=model)
        assert isinstance(model, IKerasModel)
        self._model: IKerasModel = model
        self._dataset: IDataset = dataset
        self._nb_workers = nb_workers

        self._keras_model = Model(inputs=self._model.get_inputs(), outputs=self._model.get_outputs())

        # print_info(self._model.get_inputs())
        # print_info(self._model.get_outputs())

        optimizer = self._model.get_optimizer()
        self._keras_model.compile(optimizer=optimizer, loss=self._model.get_loss(), loss_weights=[1., 1.])

        self._keras_model.summary()
        #https://github.com/tensorflow/tensorflow/issues/20999
        lr_scheduler = LearningRateScheduler(lr_decay)

        # This function keeps the learning rate at 0.001 for the first ten epochs
        # and decreases it exponentially after that.
        def scheduler(epoch):
            if epoch < 10:
                return 0.001
            else:
                return 0.001 * tf.math.exp(0.1 * (10 - epoch))

        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)


        ckpt = CustomModelCheckpoint(model=self._model,
                                     path=self._model.model_dir + '/_model-{epoch:02d}.h5',
                                     period=1, #TODO FLAGS.save_checkpoint_epochs
                                     save_weights_only=True)

        self.callbacks = [ckpt]

    def train(self, max_steps=None, num_epochs=None):
        pass

    def evaluate(self, steps=None, checkpoint_path=None):
        pass

    def train_and_evaluate(self, max_train_steps=None, eval_steps=None, num_epochs=None):
        train_data_generator = self._dataset.get_train_dataset_gen()
        history = self._keras_model.fit_generator(train_data_generator,
                                                  epochs=num_epochs,
                                                  steps_per_epoch=self._dataset.train_samples_count / self._dataset.batch_size,
                                                  workers=self._nb_workers,
                                                  use_multiprocessing=False,
                                                  max_queue_size=10,
                                                  callbacks=self._model.get_callbacks() + self.callbacks,
                                                  verbose=1)

