import tensorflow as tf
from vitaflow.datasets.datasets import get_dataset


def test_iterator(self):
    iterator = self._data_iterator.train_input_fn().make_initializable_iterator()
    training_init_op = iterator.initializer
    num_samples = self._data_iterator.num_train_examples
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

def run(dataset_name, is_tf_dataset, is_genenrator):
    """

    :param dataset_name:
    :param is_tf_dataset:
    :param is_genenrator:
    :return:
    """
    pass
