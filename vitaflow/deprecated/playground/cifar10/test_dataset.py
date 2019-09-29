import os
import tensorflow as tf
from vitaflow.deprecated.playground import Cifar10Dataset


class TestCifiar10Dataset(tf.test.TestCase):

    def setUp(self):
        self.dataset = Cifar10Dataset()

    def test_train_files(self):
        self.assertTrue(len(os.listdir(self.dataset.TRAIN_OUT_PATH)) == 5)


if __name__ == '__main__':
    tf.test.main()