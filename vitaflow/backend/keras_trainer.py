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

from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint, Callback
import numpy as np
import tensorflow as tf
from PIL import Image



def lr_decay(init_learning_rate, lr_decay_rate, lr_decay_steps, epoch):
    return init_learning_rate * np.power(lr_decay_rate, epoch // lr_decay_steps)


class CustomModelCheckpoint(Callback):
    def __init__(self, model, path, period, save_weights_only):
        super(CustomModelCheckpoint, self).__init__()
        self.period = period
        self.path = path
        # We set the model (non multi gpu) under an other name
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
                 experiment_name,
                 model: IKerasModel,
                 dataset: IDataset,
                 max_train_steps,
                 validation_interval_steps,
                 nb_workers=4,
                 stored_model=""):
        TrainerBase.__init__(self,
                             experiment_name=experiment_name,
                             max_train_steps=max_train_steps,
                             validation_interval_steps=validation_interval_steps,
                             stored_model=stored_model)
        assert isinstance(model, IKerasModel)
        self._model: IKerasModel = model
        self._dataset: IDataset = dataset
        self._nb_workers = nb_workers

        self._keras_model = Model(inputs=self._model.get_inputs(), outputs=self._model.get_outputs())

        self._keras_model.compile(optimizer=self._model.get_optimizer(), loss=self._model.get_loss(), loss_weights=[1., 1.])


        lr_scheduler = LearningRateScheduler(lr_decay)
        ckpt = CustomModelCheckpoint(model=self._model,
                                     path=self._model.model_dir + '/model-{epoch:02d}.h5',
                                     period=1, #TODO FLAGS.save_checkpoint_epochs
                                     save_weights_only=True)

        self.callbacks = [lr_scheduler, ckpt]

    def train(self, max_steps=None, num_epochs=None):
        pass

    def evaluate(self, steps=None, checkpoint_path=None):
        pass

    def train_and_evaluate(self, max_train_steps=None, eval_steps=None, num_epochs=None):
        train_data_generator = self._dataset.get_number_steps_per_epcoh(number_samples=self._dataset.train_samples_count)
        history = self._keras_model.fit_generator(train_data_generator,
                                                  epochs=num_epochs,
                                                  steps_per_epoch=self._dataset.train_samples_count / self._dataset.batch_size,
                                                  workers=self._nb_workers,
                                                  use_multiprocessing=True,
                                                  max_queue_size=10,
                                                  callbacks=self._model.get_callbacks() + self.callbacks,
                                                  verbose=1)