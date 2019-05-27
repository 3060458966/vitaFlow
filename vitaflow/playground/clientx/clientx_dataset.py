# Copyright 2018 The vitFlow Authors. All Rights Reserved.
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
CLIENTX Dataset Class
"""

__all__ = ["CLIENTXDataset"]
__version__ = '0.0.1'
__author__ = 'Gaurish Thakkar'

import os
import shutil

import gin

from vitaflow.internal.dataset_types.dataset_types import ICSVSeq2SeqType1
from vitaflow.internal.ipreprocessor import IPreprocessor
from vitaflow.utils.print_helper import *

@gin.configurable
class CLIENTXDataset(IPreprocessor, ICSVSeq2SeqType1):
    """
    Converts the given train, val, test folder to IOB format

    .. code-block:: text


    """

    def __init__(self,
                 experiment_name,
                 preprocessed_data_path,
                 experiment_root_directory,
                 train_data_path="train",
                 validation_data_path="val",
                 test_data_path="test",
                 minimum_num_words=5,
                 over_write=False
                 ):
        self._experiment_name = experiment_name
        self._preprocessed_data_path = preprocessed_data_path
        self._experiment_root_directory = experiment_root_directory
        self._train_data_path = train_data_path
        self._validation_data_path = validation_data_path
        self._test_data_path = test_data_path
        self._over_write = over_write
        self._temp_data = experiment_root_directory

        IPreprocessor.__init__(self,
                               experiment_name=experiment_name,
                               experiment_root_directory=experiment_root_directory,
                               train_data_path=train_data_path,
                               validation_data_path=validation_data_path,
                               test_data_path=test_data_path,
                               preprocessed_data_path=preprocessed_data_path,
                               )
        self._prepare_data()

    def _create_target_directories(self):
        """
        To setup destination folders structure if not present.
        :return:
        """
        if os.path.exists(self.PROCESSED_DATA_OUT_DIR):
            if self._over_write:
                print_info("Deleting data folder: {}".format(self.PROCESSED_DATA_OUT_DIR))
                shutil.rmtree(self.PROCESSED_DATA_OUT_DIR)
                print_info("Recreating data folder: {}".format(self.PROCESSED_DATA_OUT_DIR))
                os.makedirs(self.PROCESSED_DATA_OUT_DIR)
            else:
                print_info("Skipping preprocessing step, since the data might already be available")
        else:
            print_info("Creating data folder: {}".format(self.PROCESSED_DATA_OUT_DIR))
            os.makedirs(self.PROCESSED_DATA_OUT_DIR)

    def _place_dataset(self, origin_file_path, out_dir):
        """
        Converts CoNLL 2003 data set text files into CSV file for each
        example/statement.
        :param txt_file_path: Input text file path
        :param out_dir: Output directory to store CSV files
        :return: Creates files in the specified train/val/test paths
        """
        from distutils.dir_util import copy_tree

        # copy subdirectory example
        copy_tree(origin_file_path, out_dir)

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        else:
            print_info("Found previous version at {}".format(out_dir))
            return

        print_info("Writing data to {}...".format(out_dir))

    def _prepare_data(self):
        """
        Prepares the data for training
        :return:
        """
        # TODO hardcoded values need to change
        print_info("Preprocessing the train data...")
        self._place_dataset(os.path.join(self._temp_data, "train"),
                            self.TRAIN_OUT_PATH)

        print_info("Preprocessing the test data...")
        self._place_dataset(os.path.join(self._temp_data, "test"),
                            self.TEST_OUT_PATH)

        print_info("Preprocessing the validation data...")
        self._place_dataset(os.path.join(self._temp_data, "val"),
                            self.VAL_OUT_PATH)
