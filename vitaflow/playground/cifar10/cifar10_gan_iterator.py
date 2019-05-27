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
CIFAR10 Basic Iterator
"""

import os
import numpy as np
from overrides import overrides
from tqdm import tqdm

import tensorflow as tf
from tensorflow import TensorShape, Dimension

# from vitaflow.internal import HParams
from vitaflow.internal import IPreprocessor
from vitaflow.internal import IIteratorBase
from vitaflow.internal.features import GANFeature
from vitaflow.utils.os_helper import check_n_makedirs, print_info


class Cifar10GanIterator(IIteratorBase, GANFeature):
    """
    References: https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/cifar10.py
    https://cntk.ai/pythondocs/CNTK_201A_CIFAR-10_DataLoader.html
    """

    def __init__(self,experiment_root_directory,
                 experiment_name,
                 number_test_of_samples,
                 batch_size=32,
                 prefetch_size=32,
                 noise_dim=30,
                 iterator_name="Cifar10BasicIterator",
                 preprocessed_data_path="preprocessed_data",
                 train_data_path="train",
                 validation_data_path="val",
                 test_data_path="test",
                 dataset=None):
        GANFeature.__init__(self)
        IIteratorBase.__init__(self,
                               experiment_root_directory=experiment_root_directory,
                               experiment_name=experiment_name,
                               batch_size=batch_size,
                               prefetch_size=prefetch_size,
                               dataset=dataset)

        self._experiment_root_directory = experiment_root_directory
        self._experiment_name = experiment_name
        self._batch_size = batch_size
        self._prefetch_size = prefetch_size
        self._dataset = dataset
        self._iterator_name = iterator_name
        self._preprocessed_data_path = preprocessed_data_path
        self._train_data_path = train_data_path
        self._validation_data_path = validation_data_path
        self._test_data_path = test_data_path
        self._noise_dim = noise_dim
        
        # self._hparams = HParams(hparams, self.default_hparams())

        self.EXPERIMENT_ROOT_DIR = os.path.join(self._experiment_root_directory,
                                                self._experiment_name)
        self.PREPROCESSED_DATA_OUT_DIR = os.path.join(self.EXPERIMENT_ROOT_DIR,
                                                      self._preprocessed_data_path)
        self.TRAIN_OUT_PATH = os.path.join(self.PREPROCESSED_DATA_OUT_DIR,
                                           self._train_data_path)
        self.TEST_OUT_PATH = os.path.join(self.PREPROCESSED_DATA_OUT_DIR,
                                          self._test_data_path)
        self.OUT_DIR = os.path.join(self.EXPERIMENT_ROOT_DIR,
                                    self._iterator_name)

        # This rule is assumed to be correct if the previous stage is of IPreprocessor
        self.TRAIN_FILES_IN_PATH = os.path.join(self.PREPROCESSED_DATA_OUT_DIR, "train/")
        self.TEST_FILES_IN_PATH = os.path.join(self.PREPROCESSED_DATA_OUT_DIR, "test/")


        check_n_makedirs(self.OUT_DIR)

        # Width and height of each image.
        self._img_size = 32

        # Number of channels in each image, 3 channels: Red, Green, Blue.
        self._num_channels = 3

        # Length of an image when flattened to a 1-dim array.
        self._img_size_flat = self._img_size * self._img_size * self._num_channels

        # Number of classes.
        self._num_classes = 10 # TODO

        # Number of files for the training-set.
        self._num_files_train = 5

        # Number of images for each batch-file in the training-set.
        self._images_per_file = 10000

        # Total number of images in the training-set.
        # This is used to pre-allocate arrays for efficiency.
        self._num_images_train = self._num_files_train * self._images_per_file

        self._images, self._labels = self._load_training_data()
        self._images = self._images.astype("float32")

    # @staticmethod
    # def default_hparams():
    #     """
    #     .. role:: python(code)
    #        :language: python
    #
    #     .. code-block:: python
    #
    #         {
    #             "experiment_root_directory" : os.path.expanduser("~") + "/vitaFlow/",
    #             "experiment_name" : "test_experiment",
    #             "iterator_name" : "conll_data_iterator",
    #             "preprocessed_data_path" : "preprocessed_data",
    #             "train_data_path" : "train",
    #             "validation_data_path" : "val",
    #             "test_data_path" : "test",
    #             "batch_size" : 32,
    #         }
    #
    #     Here:
    #
    #     "experiment_root_directory" : str
    #         Root directory where the data is downloaded or copied, also
    #         acts as the folder for any subsequent experimentation
    #
    #     "experiment_name" : str
    #         Name for the current experiment
    #
    #     "iterator_name" : str
    #         Name of the data iterator
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
    #     "batch_size" : int
    #         Batch size for the current iterator
    #
    #     "noise_dim" : int
    #         1D Dimension of noise signal
    #
    #
    #     :return: A dictionary of hyperparameters with default values
    #     """
    #
    #     hparams = IPreprocessor.default_hparams()
    #     update(IIteratorBase.default_hparams())
    #     update({
    #         "iterator_name": "Cifar10GanIterator",
    #         "noise_dim" : 30
    #     })
    #     return hparams

    @property
    def num_labels(self):
        return self._num_classes

    @property
    def num_train_samples(self):
        return 45000

    @property
    def num_val_samples(self):
        raise 5000

    @property
    def num_test_samples(self):
        raise 10000

    def _unpickle(self, file):
        import pickle
        with open(file, 'rb') as fo:
            data_dict = pickle.load(fo, encoding='bytes')
        return data_dict

    def _normalize(self, x):
        """
        Normalize a list of sample image data in the range of 0 to 1
        : x: List of image data.  The image shape is (32, 32, 3)
        : return: Numpy array of normalize data
        """
        minV = np.min(x)
        maxV = np.max(x)
        ret = (x - minV)/maxV
        return ret

    def _convert_images(self, raw):
        """
        Convert images from the CIFAR-10 format and
        return a 4-dim array with shape: [image_number, height, width, channel]
        where the pixels are floats between 0.0 and 1.0.
        """

        # Convert the raw images from the data-files to floating-points.
        raw_float = np.array(raw, dtype=float) / 255.0

        # Reshape the array to 4-dimensions.
        images = raw_float.reshape([-1, self._num_channels, self._img_size, self._img_size])

        # Reorder the indices of the array.
        images = images.transpose([0, 2, 3, 1])

        return self._normalize(images)

    def _load_data(self, filepath):
        """
        Load a pickled data-file from the CIFAR-10 data-set
        and return the converted images (see above) and the class-number
        for each image.
        """

        # Load the pickled data-file.
        data = self._unpickle(filepath)

        # Get the raw images.
        raw_images = data[b'data']

        # Get the class-numbers for each image. Convert to numpy-array.
        classes = np.array(data[b'labels'])

        # Convert the images.
        images = self._convert_images(raw_images)

        return images, classes

    def _one_hot_encoded(self, class_numbers, num_classes=None):
        """
        Generate the One-Hot encoded class-labels from an array of integers.
        For example, if class_number=2 and num_classes=4 then
        the one-hot encoded label is the float array: [0. 0. 1. 0.]
        :param class_numbers:
            Array of integers with class-numbers.
            Assume the integers are from zero to num_classes-1 inclusive.
        :param num_classes:
            Number of classes. If None then use max(class_numbers)+1.
        :return:
            2-dim array of shape: [len(class_numbers), num_classes]
        """

        # Find the number of classes if None is provided.
        # Assumes the lowest class-number is zero.
        if num_classes is None:
            num_classes = np.max(class_numbers) + 1

        return np.eye(num_classes, dtype=float)[class_numbers]

    def _load_training_data(self):
        """
        Load all the training-data for the CIFAR-10 data-set.
        The data-set is split into 5 data-files which are merged here.
        Returns the images, class-numbers and one-hot encoded class-labels.
        """

        # Pre-allocate the arrays for the images and class-numbers for efficiency.
        images = np.zeros(shape=[self._num_images_train, self._img_size, self._img_size, self._num_channels], dtype=float)
        cls = np.zeros(shape=[self._num_images_train], dtype=int)

        # Begin-index for the current batch.
        begin = 0

        # For each data-file.
        for i in tqdm(range(self._num_files_train), desc="loading: "):
            filepath = os.path.join(self.TRAIN_OUT_PATH, "data_batch_" + str(i + 1))
            # Load the images and class-numbers from the data-file.
            images_batch, cls_batch = self._load_data(filepath=filepath)

            # Number of images in this batch.
            num_images = len(images_batch)

            # End-index for the current batch.
            end = begin + num_images

            # Store the images into the array.
            images[begin:end, :] = images_batch

            # Store the class-numbers into the array.
            cls[begin:end] = cls_batch

            # The begin-index for the next batch is the current end-index.
            begin = end

        return images, self._one_hot_encoded(class_numbers=cls, num_classes=self._num_classes)

    def _load_test_data(self):
        """
        Load all the test-data for the CIFAR-10 data-set.
        Returns the images, class-numbers and one-hot encoded class-labels.
        """
        filepath = os.path.join(self.TEST_OUT_PATH, "test_batch")
        images, cls = self._load_data(filepath=filepath)

        return images, self._one_hot_encoded(class_numbers=cls, num_classes=self._num_classes)

    def _yield_samples(self, features, labels):
        for feature, label in zip(features, labels):
            noise = np.random.random(self._noise_dim)
            yield feature, noise, label

    def _yield_train_samples(self):
        return self._yield_samples(self._images[:45000], self._labels[:45000])

    def _yield_val_samples(self):
        return self._yield_samples(self._images[:-5000], self._labels[:-5000])

    def _yield_test_samples(self):
        images, labels = self._load_test_data()
        return self._yield_samples(images, labels)

    @overrides
    def _get_train_input_fn(self):

        dataset = tf.data.Dataset.from_generator(self._yield_train_samples,
                                                 (tf.float32, tf.float32, tf.int32),
                                                 output_shapes=(TensorShape([Dimension(32), Dimension(32), Dimension(3)]),
                                                                TensorShape([Dimension(self._noise_dim)]),
                                                                TensorShape(Dimension(10))))

        dataset = dataset.map(lambda image, noise, label: ({self.FEATURE_1_NAME: image, self.FEATURE_2_NAME: noise},
                                                           label))

        dataset = dataset.batch(batch_size=self._batch_size)
        dataset = dataset.prefetch(self._prefetch_size)
        print_info("Trianing Dataset output sizes are: ")
        print_info(dataset.output_shapes)
        return dataset

    @overrides
    def _get_val_input_fn(self):

        dataset = tf.data.Dataset.from_generator(self._yield_val_samples,
                                                 (tf.float32, tf.float32, tf.int32),
                                                 output_shapes=(TensorShape([Dimension(32), Dimension(32), Dimension(3)]),
                                                                TensorShape([Dimension(self._noise_dim)]),
                                                                TensorShape(Dimension(10))))

        dataset = dataset.map(lambda image, noise, label: ({self.FEATURE_1_NAME: image, self.FEATURE_2_NAME: noise},
                                                           label))

        dataset = dataset.batch(batch_size=self._batch_size)
        dataset = dataset.prefetch(self._prefetch_size)
        print_info("Validation Dataset output sizes are: ")
        print_info(dataset.output_shapes)
        return dataset

    @overrides
    def _get_test_input_fn(self):

        dataset = tf.data.Dataset.from_generator(self._yield_test_samples,
                                                 (tf.float32, tf.float32, tf.int32),
                                                 output_shapes=(TensorShape([Dimension(32), Dimension(32), Dimension(3)]),
                                                                TensorShape([Dimension(self._noise_dim)]),
                                                                TensorShape(Dimension(10))))

        dataset = dataset.map(lambda image, noise, label: ({self.FEATURE_1_NAME: image, self.FEATURE_2_NAME: noise},
                                                           label))

        dataset = dataset.batch(batch_size=self._batch_size)
        dataset = dataset.prefetch(self._prefetch_size)
        print_info("Dataset output sizes are: ")
        print_info(dataset.output_shapes)
        return dataset