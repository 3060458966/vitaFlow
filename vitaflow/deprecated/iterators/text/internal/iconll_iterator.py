# # Copyright 2018 The vitFlow Authors. All Rights Reserved.
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
# Data iterator interface for CoNLL data sets
# """
#
# from overrides import overrides
# import tensorflow as tf
#
# from vitaflow.config.hyperparams import HParams
# from vitaflow.data.internal import IPreprocessor
# from vitaflow.data.internal.iterator_base import IIteratorBase
# from vitaflow.data.text.vocabulary import SpecialTokens
# from vitaflow.helpers.print_helper import *
# from vitaflow.helpers.os_helper import check_n_makedirs
#
# __all__ = [
#     "ICoNLLIterator"
# ]
#
# class ICoNLLIterator(IIteratorBase):
#     def __init__(self, hparams=None, dataset=None):
#         '''
#         Data Iterators with different features type are expected to
#         implement this interface, exposing the input functions and their hooks
#         :param experiment_dir:
#         :param batch_size:
#
#         '''
#         IIteratorBase.__init__(self, hparams=hparams, dataset=dataset)
#         self._hparams = HParams(hparams, self.default_hparams())
#
#         #
#         # def _init(self):
#
#         self.EXPERIMENT_ROOT_DIR = self._hparams.experiment_root_directory + \
#                                    "/" + self._hparams.experiment_name
#
#         self.DATA_OUT_DIR = self.EXPERIMENT_ROOT_DIR + "/" + \
#                             self._hparams.preprocessed_data_path + "/"
#
#         self.TRAIN_OUT_PATH = self.DATA_OUT_DIR + "/" + \
#                               self._hparams.train_data_path + "/"
#         self.VAL_OUT_PATH = self.DATA_OUT_DIR + "/" + \
#                             self._hparams.validation_data_path + "/"
#         self.TEST_OUT_PATH = self.DATA_OUT_DIR + "/" + \
#                              self._hparams.test_data_path + "/"
#
#         self.OUT_DIR = self.EXPERIMENT_ROOT_DIR + "/" + self._hparams.name + "/"
#
#         # This rule is assumed to be correct if the previous stage is of IPreprocessor
#         self.TRAIN_FILES_IN_PATH = self.DATA_OUT_DIR + "/train/"
#         self.VAL_FILES_IN_PATH = self.DATA_OUT_DIR + "/val/"
#         self.TEST_FILES_IN_PATH = self.DATA_OUT_DIR + "/test/"
#
#
#         self.WORDS_VOCAB_FILE = self.OUT_DIR + "/" + self._hparams.text_col + "_" + "vocab.tsv"
#         self.CHARS_VOCAB_FILE = self.OUT_DIR + "/" + self._hparams.text_col + "_" + "chars_vocab.tsv"
#         self.ENTITY_VOCAB_FILE = self.OUT_DIR + "/" + self._hparams.entity_col + "_vocab.tsv"
#
#         check_n_makedirs(self.OUT_DIR)
#
#         self.padded_shapes = (tf.TensorShape([None]),   # sentence of unknown size
#                               tf.TensorShape([None]),
#                               tf.TensorShape([None]))  # labels of unknown size
#
#         self.padding_values = (SpecialTokens.PAD_WORD,
#                                SpecialTokens.PAD_CHAR,# sentence padded on the right with id_pad_word
#                                SpecialTokens.PAD_TAG)    # labels padded on the right with id_pad_tag
#
#     @staticmethod
#     def default_hparams():
#         """
#         .. role:: python(code)
#            :language: python
#
#         .. code-block:: python
#
#             {
#             "name" : "conll_data_iterator",
#             "experiment_root_directory" : "default_experiment",
#             "preprocessed_data_path" : "preprocessed_data",
#             "train_data_path" : "train",
#             "validation_data_path" : "val",
#             "test_data_path" : "test",
#
#             "text_col" : 0,
#             "entity_col" : 3,
#             "batch_size" : 16,
#             }
#
#         Here:
#
#         "name" : str
#             Name of the data iterator
#
#         "experiment_root_directory" : str
#             Root directory where the data is downloaded or copied, also
#             acts as the folder for any subsequent experimentation
#
#         "preprocessed_data_path" : str
#             Folder path under `experiment_root_directory` where the preprocessed data
#             should be stored
#
#         "train_data_path" : str
#             Folder path under `experiment_root_directory` where the train data is stored
#
#         "validation_data_path" : str
#             Folder path under `experiment_root_directory` where the validation data is stored
#
#         "test_data_path" : str
#             Folder path under `experiment_root_directory` where the test data is stored
#
#         "text_col" : int
#             Text column to be referred in the preprocessed CoNLL data CSV files
#
#         "entity_col" : int
#             Entity/Label column to be referred in the preprocessed CoNLL data CSV files
#
#         :return: A dictionary of hyperparameters with default values
#         """
#
#         hparams = IPreprocessor.default_hparams()
#
#         hparams.update({
#             "name" : "conll_data_iterator",
#             "text_col" : "0",
#             "entity_col" : "3",
#         })
#
#         hparams.update(IIteratorBase.default_hparams())
#
#         return hparams
#
#     @property
#     def _train_files_path(self):
#         raise NotImplementedError
#
#     @property
#     def _val_files_path(self):
#         raise NotImplementedError
#
#
#     def _user_map_func(self, csv_file_path):
#         raise NotImplementedError
#
#     def _user_resize_func(self, feat1, feat2, label):
#         raise NotImplementedError
#
#     @overrides
#     def _get_train_input_fn(self):
#         dataset = tf.data.Dataset.from_tensor_slices((self._train_files_path))
#         dataset = dataset.map(
#             lambda filename: tuple(tf.py_func(
#                 self._user_map_func, [filename], [(tf.string, tf.string), tf.string])),
#             num_parallel_calls=2)
#         # dataset = dataset.shuffle(len(self._train_labels))
#         # dataset.prefetch()
#         # dataset = dataset.repeat(self._num_epochs)
#         # dataset = dataset.map(self._user_resize_func)
#         # dataset = dataset.prefetch(self._hparams.batch_size*2)
#         # Create batches and pad the sentences of different length
#
#         dataset = dataset. \
#             padded_batch(self._hparams.batch_size,
#                          padded_shapes=self.padded_shapes,
#                          padding_values=self.padding_values)
#         # dataset = dataset.batch(batch_size=self._hparams.batch_size)
#         print_info("Dataset output sizes are: ")
#         print_info(dataset.output_shapes)
#
#
#         return dataset
#
#     @overrides
#     def _get_val_input_fn(self):
#         dataset = tf.data.Dataset.from_tensor_slices((self._val_files_path))
#         dataset = dataset.map(
#             lambda filename: tuple(tf.py_func(
#                 self._user_map_func, [filename], [tf.string, tf.string])),
#             num_parallel_calls=4)
#         # dataset = dataset.shuffle(len(self._val_labels))
#         # dataset = dataset.repeat(self._num_epochs)
#         dataset = dataset.map(self._user_resize_func, num_parallel_calls=4)
#
#         dataset = dataset.prefetch(self._hparams.batch_size*2)
#         dataset = (dataset
#                    .shuffle(buffer_size=self._hparams.batch_size)
#                    .padded_batch(self._hparams.batch_size,
#                                  padded_shapes=self.padded_shapes,
#                                  padding_values=self.padding_values)
#                    )
#
#         print_info("Dataset output sizes are: ")
#         print_info(dataset.output_shapes)
#
#         return dataset
#
#
#     @overrides
#     def _get_test_input_function(self):
#         raise NotImplementedError
