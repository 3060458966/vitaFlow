import threading
from tqdm import tqdm
from vitaflow.utils.print_helper import print_info


class ThreadsafeIter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self): # Python 3
        with self.lock:
            return next(self.it)

    def next(self): # Python 2
        with self.lock:
            return self.it.next()


def thread_safe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return ThreadsafeIter(f(*a, **kw))
    return g


class IDataset(object):
    def __init__(self,
                 data_in_dir,
                 data_out_dir,
                 batch_size,
                 is_preprocess,
                 num_cores=4):
        self._data_in_dir = data_in_dir
        self._data_out_dir = data_out_dir
        self._batch_size = batch_size
        self._is_preprocess = is_preprocess
        self._num_cores = num_cores

        self._num_train_samples = None
        self._num_val_samples = None
        self._num_test_samples = None

    @property
    def train_samples_count(self):
        if self._num_train_samples is None:
            self._num_train_samples = self._get_num_train_samples()
            # TODO store the value in disk
        return self._num_train_samples

    @property
    def val_samples_count(self):
        if self._num_val_samples is None:
            self._num_val_samples = self._get_num_val_samples()
            # TODO store the value in disk
        return self._num_val_samples

    @property
    def test_samples_count(self):
        if self._num_test_samples is None:
            self._num_test_samples = self._get_num_test_samples()
            # TODO store the value in disk
        return self._num_test_samples

    @property
    def batch_size(self):
        return self._batch_size

    def dataset_to_iterator(self, dataset):
        # Create an iterator
        iterator = dataset.make_one_shot_iterator()

        # Create your tf representation of the iterator
        features, label = iterator.get_next()

        # TODO reshape
        # # Bring your picture back in shape
        # image = tf.reshape(image, [-1, 256, 256, 1])
        #
        # # Create a one hot array for your labels
        # label = tf.one_hot(label, NUM_CLASSES)

        return features, label

    def get_number_steps_per_epcoh(self, number_samples):
        res = number_samples // self._batch_size
        print("\n\n\n\n\n")
        print_info(">"*50)
        print_info(f"Number of examples per epoch is {number_samples}")
        print_info(f"Batch size is {self._batch_size}")
        print_info(f"Number of steps per epoch is {res}")
        print_info(">" * 50)
        print("\n\n\n\n\n")
        return res

    def get_train_dataset_gen(self, num_epochs=None):
        """
        Returns an data set generator function that can be used in train loop
        :return:
        """
        raise NotImplementedError("No implementation for general Dataset Iterator")

    def get_val_dataset_gen(self, num_epochs=None):
        """
        Returns an data set generator function that can be used in validation loop
        :return:
        """
        raise NotImplementedError("No implementation for general Dataset Iterator")

    def get_test_dataset_gen(self):
        """
        Returns an data set generator function that can be used in test loop
        :return:
        """
        raise NotImplementedError("No implementation for general Dataset Iterator")

    def get_serving_dataset(self, file_or_path):
        raise NotImplementedError("No implementation for general Dataset Iterator")

    def get_tf_train_dataset(self):
        raise NotImplementedError("No implementation for TensorFlow Dataset API")

    def get_tf_val_dataset(self):
        raise NotImplementedError("No implementation for TensorFlow Dataset API")

    def get_tf_test_dataset(self):
        raise NotImplementedError("No implementation for TensorFlow Dataset API")

    def get_tf_serving_dataset(self, file_or_path):
        raise NotImplementedError("No implementation for TensorFlow Dataset API")

    def _get_num_train_samples(self):
        raise NotImplementedError("Respective inherited class should implement this routine")

    def _get_num_val_samples(self):
        raise NotImplementedError("Respective inherited class should implement this routine")

    def _get_num_test_samples(self):
        raise NotImplementedError("Respective inherited class should implement this routine")

    def test_tf_dataset(self):
        i = 0
        for features, label in tqdm(self.get_tf_train_dataset()):
            for key in features.keys():
                print("Batch {} =>  Shape of feature : {} is {}".format(i, key, features[key].shape))
                i = i + 1