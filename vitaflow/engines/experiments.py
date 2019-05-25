# Copyright 2018 The Shabda Authors. All Rights Reserved.
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
Experiments class that allows easy plug n play of modules
"""
import logging
import os
import shutil
import time
from importlib import import_module

import gin

import tensorflow as tf
from tqdm import tqdm

from vitaflow.utils.hyperparams import HParams
from vitaflow.utils.print_helper import *
from vitaflow.engines.executor import Executor

# get TF logger
log = logging.getLogger('tensorflow')
log.setLevel(logging.DEBUG)

CGREEN2 = '\33[92m'
CEND = '\33[0m'


@gin.configurable
class Experiments(object):
    """
    Experiments uses dataset, data iterator & model factory classes and import them
    dynamically based on the string.
    This allows the user to choose the modules dynamically and run the experiments without ever writing the
    code when we need mix and experiment dataset and modules.
    """

    def __init__(self,
                 num_epochs=5,
                 dataset_class_with_path=None,
                 iterator_class_with_path=None,
                 model_class_with_path=None,
                 save_checkpoints_steps=50,
                 keep_checkpoint_max=5,
                 save_summary_steps=25,
                 log_step_count_steps=10,
                 clear_model_data=False,
                 plug_dataset=True,
                 mode='train',
                 batch_size=8):
        
        self.mode = mode
        self._dataset = None
        self.data_iterator = None
        self._model = None

        self.num_epochs = num_epochs
        self.dataset_class_with_path = dataset_class_with_path
        self.iterator_class_with_path = iterator_class_with_path
        self.model_class_with_path = model_class_with_path
        self.save_checkpoints_steps = save_checkpoints_steps
        self.keep_checkpoint_max = keep_checkpoint_max
        self.save_summary_steps = save_summary_steps
        self.log_step_count_steps = log_step_count_steps
        self.clear_model_data = clear_model_data
        self.plug_dataset = plug_dataset
        self.batch_size = batch_size

    def _get_class(self, package, name):
        """
        Import the givenpackage and the class dynmaically
        :param package: Pacakage path of the class
        :param name: Name of the class
        :return: Instance of the class object
        """
        return getattr(import_module(package), name)

    def get_dataset_reference(self, dataset_class_with_path):
        """
        Uses the dataset name to get the reference from the dataset factory class
        :param dataset_class_with_path:
        :return:
        Eg: vitaflow.data.text.conll.conll_2003_dataset.CoNLL2003Dataset
        """

        print_debug("Dynamically importing dataset : " + dataset_class_with_path)
        package, name = dataset_class_with_path.rsplit(".", 1)
        # dataset = DatasetFactory.get(dataset_file_name=dataset_name)
        dataset = self._get_class(package=package, name=name)
        return dataset
    
    def get_iterator_reference(self, iterator_class_with_path):
        """
        Uses the iterator name to get the reference from the iterator factory class
        :param iterator_class_with_path:
        :return:
        """

        print_debug("Dynamically importing iterator : " + iterator_class_with_path)
        # iterator = DataIteratorFactory.get(iterator_name=iterator_name)
        package, name = iterator_class_with_path.rsplit(".", 1)
        iterator = self._get_class(package=package, name=name)
        return iterator

    def get_model_reference(self, model_class_with_path):
        """
        Uses the model name to get the reference from the model factory class
        :param model_class_with_path:
        :return:
        """

        print_debug("Dynamically importing model : " + model_class_with_path)
        package, name = model_class_with_path.rsplit(".", 1)
        model = self._get_class(package=package, name=name)
        # model = ModelsFactory.get(model_name=model_name)
        return model

    def check_interoperability_n_import(self):
        # Using factory classes get the handle for the actual classes from string
        if self.plug_dataset:
            self._dataset = self.get_dataset_reference(self.dataset_class_with_path)
        self._data_iterator = self.get_iterator_reference(self.iterator_class_with_path)
        self._model = self.get_model_reference(self.model_class_with_path)

        # if not self._data_iterator.dataset_type == self._dataset.dataset_type:
        #     print_info("Possible data iterators are: {}".
        #                format(DataIteratorFactory.get_supported_data_iterators(self._dataset.dataset_type)))
        #     raise RuntimeError("Selected data iterator and data set can't be used together")
        #
        # if not self._model.data_iterator_type == self._data_iterator.data_iterator_type:
        #     print_info("Possible models are: {}".
        #                format(ModelsFactory.get_supported_data_iterators(self._dataset.dataset_type)))
        #     raise RuntimeError("Selected model and data iterator can't be used together")

    def _init_tf_config(self):
        run_config = tf.ConfigProto()
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

    def setup(self):
        self.check_interoperability_n_import()
        # Initialize the handles and call any user specific init() methods
        if self.plug_dataset:
            self._dataset = self._dataset()
        #TODO avoid loading train data while prediction
        self._data_iterator = self._data_iterator(dataset=self._dataset)
        print_error(self._model)
        self._model = self._model(data_iterator=self._data_iterator)

    def test_iterator(self):
        iterator = self._data_iterator.train_input_fn().make_initializable_iterator()
        training_init_op = iterator.initializer
        num_samples = self._data_iterator.num_train_samples
        next_element = iterator.get_next()
        batch_size = self.batch_size

        with tf.Session() as sess:
            sess.run(training_init_op)
            start_time = time.time()

            pbar = tqdm(desc="steps", total=num_samples)

            i = 0
            while (True):
                res = sess.run(next_element)
                pbar.update()
                try:
                    if True:
                        print("Data shapes : ", end=" ")
                        for key in res[0].keys():
                            print(res[0][key].shape, end=", ")
                        print(" label shape : {}".format(res[1].shape))

                except tf.errors.OutOfRangeError:
                    break
            end_time = time.time()

            print_debug("time taken is {} ".format(end_time - start_time))

        exit(0)


    
    def run(self, args):
        self.setup()
        num_samples = self._data_iterator.num_train_samples
        print_info("Number of trianing samples : {}".format(num_samples))
        batch_size = self.batch_size
        num_epochs = self.num_epochs
        mode = self.mode
        self._init_tf_config()

        if mode == "test_iterator":
            self.test_iterator()

        executor = Executor(model=self._model, data_iterator=self._data_iterator, config=self._run_config)

        if mode in ["train", "retrain"]:
            for current_epoch in tqdm(range(num_epochs), desc="Epoch"):
                current_max_steps = (num_samples // batch_size) * (current_epoch + 1)
                print("\n\n Training for epoch {} with steps {}\n\n".format(current_epoch, current_max_steps))
                executor.train(max_steps=None)
                print("\n\n Evaluating for epoch\n\n", current_epoch)
                executor.evaluate(steps=200)

        elif mode == "predict":
            self._data_iterator.predict_on_test_files(executor=executor)

        elif mode == "predict_instance":
            self._data_iterator.predict_on_instance(executor=executor, file_path=args.test_file_path)
        else:
            print_error("Given mode is not avaialble!")

