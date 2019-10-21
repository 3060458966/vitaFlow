
import gin
import numpy as np
import os
import random
import shutil
import tensorflow as tf
import torch
from vitaflow.utils.print_helper import memory_usage_psutil
from vitaflow.datasets.datasets import get_dataset
from vitaflow.models.models import get_model
from vitaflow.models.interface_model import IEstimatorModel, ITorchModel, IKerasModel
from vitaflow.backend.torch_trainer import TorchTrainer
from vitaflow.backend.keras_trainer import KerasTrainer
from vitaflow.backend.tf_executor import TFExecutor


@gin.configurable
class VitaFlowEngine(object):
    """
    VitaFlowEngine is a wrapper that takes the name of dataset and _model.
    Based on the _model type it selects Tensorflow / Keras / Torch training backend.
    """

    def __init__(self,
                 experiment_name=gin.REQUIRED,
                 dataset_name=gin.REQUIRED,
                 model_name=gin.REQUIRED,
                 experiment_root_dir=gin.REQUIRED,
                 num_epochs=None,
                 num_max_steps=None,
                 validation_interval_steps=None,
                 store_model_epoch_interval=None,
                 store_model_interval_steps=None,
                 keep_checkpoint_max=None,
                 save_summary_steps=None,
                 log_step_count_steps=None,
                 max_steps_without_decrease=None,
                 random_seed=42):
        """

        :param experiment_name: Name of the experiment
        :param dataset_name: Dataset name. `run vitalfow/bin/vf_print_registry.py` to get the names
        :param model_name: Model name. `run vitalfow/bin/vf_print_registry.py` to get the names
        :param num_epochs: Number of epochs
        :param num_max_steps: Number of training steps. (Number of training samples / batch size * number oof epochs)
        :param validation_interval_steps: Number of steps to pass before validating the models each time
        :param store_model_interval_steps: Number of steps to pass before each time the _model is stored
        :param keep_checkpoint_max: Number of stored models to keep in disk
        :param save_summary_steps: Summary logging interval steps
        :param log_step_count_steps:
        :param max_steps_without_decrease: Number of steps to continue without any significant loss change
        :param random_seed: Seed for random generator
        """

        """ Seed and GPU setting """
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)

        self._experiment_name = experiment_name

        self._num_epochs = num_epochs
        self._num_max_steps = num_max_steps

        self._dataset = get_dataset(dataset_name)
        self._model = get_model(model_name)

        self.save_checkpoints_steps = store_model_interval_steps
        self.keep_checkpoint_max = keep_checkpoint_max
        self.save_summary_steps = save_summary_steps
        self.log_step_count_steps = log_step_count_steps
        self.max_steps_without_decrease = max_steps_without_decrease
        self._validation_interval_steps = validation_interval_steps
        self._experiment_root_dir = experiment_root_dir
        self._store_model_epoch_interval = store_model_epoch_interval

        # Initialize the dataset and _model
        self._dataset = self._dataset()
        self._model = self._model(dataset=self._dataset)

    def _init_tf_config(self):
        run_config = tf.compat.v1.ConfigProto()
        run_config.gpu_options.allow_growth = True
        # run_config.gpu_options.per_process_gpu_memory_fraction = 0.50
        run_config.allow_soft_placement = True
        run_config.log_device_placement = False
        model_dir = self._model.model_dir

        if self.clear_model_data:
            if os.path.exists(model_dir):
                shutil.rmtree(model_dir)

        self._run_config = tf.estimator.RunConfig(session_config=run_config,
                                                  save_checkpoints_steps=self.save_checkpoints_steps,
                                                  keep_checkpoint_max=self.keep_checkpoint_max,
                                                  save_summary_steps=self.save_summary_steps,
                                                  model_dir=model_dir,
                                                  log_step_count_steps=self.log_step_count_steps)
        return run_config

    def run(self, mode=None):

        if mode == "test_iterator":
            self.test_iterator()
        else:
            if isinstance(self._model, ITorchModel):
                self._model.compile(num_epochs=self._num_epochs)
                model_store_path = os.path.join(self._experiment_root_dir, self._experiment_name, self._model.name)
                executor = TorchTrainer(model=self._model,
                                        dataset=self._dataset,
                                        model_store_path=model_store_path)
                executor.train(num_max_steps=self._num_max_steps,
                               num_epochs=self._num_epochs,
                               store_model_epoch_interval=self._store_model_epoch_interval)

            elif isinstance(self._model, IEstimatorModel):
                self._init_tf_config()
                executor = TFExecutor(model=self._model,
                                      dataset=self._dataset,
                                      config=self._run_config,
                                      max_steps_without_decrease=self.max_steps_without_decrease)
                memory_usage_psutil()
                executor.train(num_epochs=self._num_epochs)
                executor.evaluate(steps=None)

            elif isinstance(self._model, IKerasModel):
                trainier = KerasTrainer(model=self._model,
                                        dataset=self._dataset,
                                        model_store_path="",
                                        nb_workers=4)
                trainier.train_and_evaluate(max_train_steps=None, eval_steps=None, num_epochs=self._num_epochs)
