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
Cifar10 Dataset Class
"""

import os
import shutil

import numpy as np
import gin
# from vitaflow.internal import HParams
from vitaflow.internal import IPreprocessor
from vitaflow.utils.print_helper import print_info
from vitaflow.utils.data_io import maybe_download
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from vitaflow.utils import registry


@registry.register_problem
@gin.configurable
class Cifar10Dataset(IPreprocessor):
    """
    Cifar10 Dataset Downloader and preprocessor
    Link: https://www.cs.toronto.edu/~kriz/cifar.html
    """
    def __init__(self, 
                 experiment_name,
                 preprocessed_data_path,
                 experiment_root_directory=os.path.join(os.path.expanduser("~"), "vitaFlow/"),
                 train_data_path="train",
                 validation_data_path="val",
                 test_data_path="test",
                 over_write=False):
        """
        """

        self._experiment_name = experiment_name 
        self._preprocessed_data_path = preprocessed_data_path
        self._experiment_root_directory = experiment_root_directory
        self._train_data_path = train_data_path
        self._validation_data_path = validation_data_path
        self._test_data_path = test_data_path
        self._over_write = over_write
        
        IPreprocessor.__init__(self,
                               experiment_name=experiment_name,
                               preprocessed_data_path=preprocessed_data_path,
                               experiment_root_directory=experiment_root_directory,
                               train_data_path=train_data_path,
                               validation_data_path=validation_data_path,
                               test_data_path=test_data_path)

        self._download_path = os.path.join(
            self._experiment_root_directory,
            self._experiment_name,
            "raw_data/"
        )
        self._link = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

        maybe_download(urls=self._link,
                       path=self._download_path,
                       filenames="cifar-10-python.tar.gz",
                       extract=True)

        self._prepare_data()


    # @staticmethod
    # def default_hparams():
    #     """
    #     .. role:: python(code)
    #        :language: python
    # 
    #     .. code-block:: python
    # 
    #         {
    #             "experiment_root_directory" : os.path.expanduser("~") + "/vitaFlow/" ,
    #             "experiment_name" : "Cifiar10Dataset",
    #             "preprocessed_data_path" : "preprocessed_data",
    #             "train_data_path" : "train",
    #             "validation_data_path" : "val",
    #             "test_data_path" : "test",
    #             "over_write" : False,
    #         }
    # 
    #     Here:
    # 
    #     "experiment_root_directory" : str
    #         Root directory where the data is downloaded or copied, also
    #         acts as the folder for any subsequent experimentation
    # 
    #     "experiment_name" : str
    #         Name of the data set
    # 
    #     "preprocessed_data_path" : str
    #         Folder path under `experiment_root_directory` where the preprocessed data
    #         should be stored
    # 
    #     "train_data_path" : str
    #         Folder path under `experiment_root_directory` where the train data is stored
    # 
    #     "validation_data_path" : str
    #         Folder path under `experiment_root_directory` where the validation data is stored
    # 
    #     "test_data_path" : str
    #         Folder path under `experiment_root_directory` where the test data is stored
    # 
    #     "over_write" : boolean
    #         Flag to over write the previous copy of the downloaded data
    # 
    #     :return: A dictionary of hyperparameters with default values
    #     """
    #     hparams = IPreprocessor.default_hparams()
    # 
    #     update({
    #         "experiment_name": "Cifiar10Dataset",
    #         "over_write" : False
    #     })
    # 
    #     return hparams

    def _create_target_directories(self):
        """
        To setup destination folders structure if not present.
        :return:
        """
        if os.path.exists(self.PREPROCESSED_DATA_OUT_DIR):
            if self._over_write:
                print_info("Deleting data folder: {}".format(self.PREPROCESSED_DATA_OUT_DIR))
                shutil.rmtree(self.PREPROCESSED_DATA_OUT_DIR)
                print_info("Recreating data folder: {}".format(self.PREPROCESSED_DATA_OUT_DIR))
                os.makedirs(self.PREPROCESSED_DATA_OUT_DIR)
            else:
                print_info("Skipping preprocessing step, since the data might already be available")
        else:
            print_info("Creating data folder: {}".format(self.PREPROCESSED_DATA_OUT_DIR))
            os.makedirs(self.PREPROCESSED_DATA_OUT_DIR)

    def copy(self, in_path, out_dir):
        path, file_name = os.path.split(in_path)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
            print_info("Copying the file {} to {}".format(in_path, out_dir))
            shutil.copy(src=in_path, dst=out_dir)
        else:
            if not os.path.exists(os.path.join(out_dir, file_name)):
                print_info("Copying the file {} to {}".format(in_path, out_dir))
                shutil.copy(src=in_path, dst=out_dir)
            else:
                print_info("Found previous copy @ {}".format(os.path.join(out_dir, file_name)))

    def _prepare_data(self):
        """Inheriting class must implement this class.
        Which not limited to, but includes:
        - Downloading or copying the data
        - Pre processing the data
        - Storing the preprocessed data as train/val/test
        :return:
        """
        #TODO avoid loading train data while prediction
        for file in ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5", "batches.meta"]:
            in_path = os.path.join(self._download_path, "cifar-10-batches-py", file)
            self.copy(in_path=in_path, out_dir=self.TRAIN_OUT_PATH)
        for file in ["test_batch", "batches.meta"]:
            in_path = os.path.join(self._download_path, "cifar-10-batches-py", file)
            self.copy(in_path=in_path, out_dir=self.TEST_OUT_PATH)

    def _load_label_names(self):
        """
        Load the label names from file
        """
        return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    def display_image_predictions(self, features, labels, predictions):
        n_classes = 10
        label_names = self._load_label_names()
        label_binarizer = LabelBinarizer()
        label_binarizer.fit(range(n_classes))
        label_ids = label_binarizer.inverse_transform(np.array(labels))

        fig, axies = plt.subplots(nrows=4, ncols=2)
        fig.tight_layout()
        fig.suptitle('Softmax Predictions', fontsize=20, y=1.1)

        n_predictions = 3
        margin = 0.05
        ind = np.arange(n_predictions)
        width = (1. - 2. * margin) / n_predictions

        for image_i, (feature, label_id, pred_indicies, pred_values) in enumerate(zip(features, label_ids, predictions.indices, predictions.values)):
            pred_names = [label_names[pred_i] for pred_i in pred_indicies]
            correct_name = label_names[label_id]

            axies[image_i][0].imshow(feature)
            axies[image_i][0].set_title(correct_name)
            axies[image_i][0].set_axis_off()

            axies[image_i][1].barh(ind + margin, pred_values[::-1], width)
            axies[image_i][1].set_yticks(ind + margin)
            axies[image_i][1].set_yticklabels(pred_names[::-1])
            axies[image_i][1].set_xticks([0, 0.5, 1.0])

        plt.show()



"""
TODO:
- https://cntk.ai/pythondocs/CNTK_201A_CIFAR-10_DataLoader.html

"""