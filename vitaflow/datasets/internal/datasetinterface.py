# # Copyright 2018 The vitaFlow Authors. All Rights Reserved.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #      http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
#
# """
# Preprocessor interface for data sets
# """
# import pickle
#
# from vitaflow.utils.print_helper import print_info
#
# __all__ = ["DatasetInterface"]
# __version__ = '0.0.1'
# __author__ = 'Mageswaran Dhandapani <mageswaran1989@gmail.com>'
#
# import os
# import gin
#
# from vitaflow.utils.hyperparams import HParams
#
# @gin.configurable
# class DatasetInterface:
#     """
#     A pre-processing interface that every dataset class should inherit.
#     This make sure that data of concern are downloaded and preprocessed alike across
#     all the data sets.
#
#     Some rules are enforced here, such that it follows following directory structure:
#
#     :param hparams (vitaflow.core.HParams): See :func:`default_hparams()`
#
#     .. code-block:: text
#
#         default_experiment/
#             preprocessed_data/
#                 train/
#                 val/
#                 test/
#
#     """
#     def __init__(self,
#                  experiment_name,
#                  preprocessed_data_path,
#                  experiment_root_directory=os.path.join(os.path.expanduser("~"), "vitaFlow/"),
#                  train_data_path = "train",
#                  validation_data_path = "val",
#                  test_data_path = "test"):
#         """
#         """
#
#         self._experiment_name = experiment_name
#         self._preprocessed_data_path = preprocessed_data_path
#         self._experiment_root_directory = experiment_root_directory
#         self._train_data_path = train_data_path
#         self._validation_data_path = validation_data_path
#         self._test_data_path = test_data_path
#
#         self.EXPERIMENT_ROOT_DIR = os.path.join(self._experiment_root_directory,
#                                                 self._experiment_name)
#
#         self.TRAIN_IN_PATH = os.path.join(self.EXPERIMENT_ROOT_DIR,
#                                           "raw_data",
#                                           self._train_data_path)
#         self.VAL_IN_PATH = os.path.join(self.EXPERIMENT_ROOT_DIR,
#                                         "raw_data",
#                                         self._validation_data_path)
#         self.TEST_IN_PATH = os.path.join(self.EXPERIMENT_ROOT_DIR,
#                                          "raw_data",
#                                          self._test_data_path)
#
#         self.PREPROCESSED_DATA_OUT_DIR = os.path.join(self.EXPERIMENT_ROOT_DIR,
#                                                       self._preprocessed_data_path)
#         self.TRAIN_OUT_PATH = os.path.join(self.PREPROCESSED_DATA_OUT_DIR,
#                                            self._train_data_path)
#         self.VAL_OUT_PATH = os.path.join(self.PREPROCESSED_DATA_OUT_DIR,
#                                          self._validation_data_path)
#         self.TEST_OUT_PATH = os.path.join(self.PREPROCESSED_DATA_OUT_DIR,
#                                           self._test_data_path)
#
#     # @staticmethod
#     # def default_hparams():
#     #     """
#     #     .. role:: python(code)
#     #        :language: python
#     #
#     #     .. code-block:: python
#     #
#     #         {
#     #             "experiment_root_directory" : os.path.expanduser("~") + "/vitaFlow/",
#     #             "experiment_name" : "experiment_name",
#     #             "preprocessed_data_path" : "preprocessed_data",
#     #             "train_data_path" : "train",
#     #             "validation_data_path" : "val",
#     #             "test_data_path" : "test"
#     #         }
#     #
#     #     Here:
#     #
#     #     "experiment_root_directory" : str
#     #         Root directory where the data is downloaded or copied, also
#     #         acts as the folder for any subsequent experimentation
#     #
#     #     "experiment_name" : str
#     #         Name for the current experiment
#     #
#     #     "preprocessed_data_path" : str
#     #         Folder path under `experiment_root_directory` where the preprocessed data
#     #         should be stored
#     #
#     #     "train_data_path" : str
#     #         Folder path under `experiment_root_directory` where the train data is stored
#     #
#     #     "validation_data_path" : str
#     #         Folder path under `experiment_root_directory` where the validation data is stored
#     #
#     #     "test_data_path" : str
#     #         Folder path under `experiment_root_directory` where the test data is stored
#     #
#     #
#     #     :return: A dictionary of hyperparameters with default values
#     #     """
#     #     return {
#     #         "experiment_root_directory": os.path.join(os.path.expanduser("~"), "vitaFlow/"),
#     #         "experiment_name": "experiment_name",
#     #         "preprocessed_data_path": "preprocessed_data",
#     #         "train_data_path": "train",
#     #         "validation_data_path": "val",
#     #         "test_data_path": "test"
#     #     }
#
#     @property
#     def dataset_dir(self):
#         """
#         Returns iterator directory `experiment_root_directory`/`experiment_name`/`iterator_name`
#         :return:
#         """
#         path = os.path.join(self._experiment_root_directory,
#                             self._experiment_name,
#                             type(self).__name__)
#         if not os.path.exists(path):
#             os.makedirs(path)
#         return path
#
#     def _create_target_directories(self):
#         """
#         Inheriting class must take care of creating experiment folder and all needed
#         sub directories to store the preprocessed training and test data
#         :return: None
#         """
#         raise NotImplementedError
#
#     def _prepare_data(self):
#         """
#         Inheriting class must implement this class.
#         Which not limited to, but includes: Downloading or copying the data, Preprocessing the data,
#         Storing the preprocessed data as train/val/test
#         :return: None
#         """
#         raise NotImplementedError
#
#     def preprocess_prepare(self):
#         """
#         Pre-process and prepares the data for training
#         :return: None
#         """
#         self._create_target_directories()
#         self._prepare_data()
#
#     def store_as_pickle(self, data, file_name):
#         file_path = os.path.join(self.dataset_dir, file_name)
#         print_info("Writing the pickle file {}...".format(file_path))
#         with open(file_path, 'wb') as f:
#             pickle.dump(data, f)
#         return None
#
#     def read_pickle(self, file_name):
#         file_path = os.path.join(self.dataset_dir, file_name)
#         if os.path.exists(file_path):
#             print_info("Reading the pickle file {}...".format(file_path))
#             with open(file_path, 'rb') as f:
#                 data = pickle.load(f)
#             return data
#         else:
#             return None
