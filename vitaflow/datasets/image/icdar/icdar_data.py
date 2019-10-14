#!/usr/bin/env python
# coding: utf-8
import glob
import multiprocessing

from absl import logging

from vitaflow.datasets.interface_dataset import IDataset
from vitaflow.datasets.image.icdar.icdar_utils import *

from vitaflow.utils.print_helper import print_error, print_info, memory_usage_psutil, print_debug
from vitaflow.utils import registry


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _mat_feature(mat):
    return tf.train.Feature(float_list=tf.train.FloatList(value=mat.flatten()))


def get_tf_records_count(files):
    total_records = 0
    for file in tqdm(files, desc="tfrecords size: "):
        total_records += 1#sum(1 for _ in tf.data.TFRecordDataset(file))
    return total_records


@gin.configurable
@registry.register_dataset
class ICDARDataset(IDataset):
    """
    Reads ICDAR 2019 dataset which is organized as train/val/test folder which contains image and
    text files with polygon co-ordinates
    References:
        https://www.geeksforgeeks.org/multiprocessing-python-set-1/
        https://www.geeksforgeeks.org/multiprocessing-python-set-2/
    """
    def __init__(self,
                 data_in_dir=gin.REQUIRED,
                 data_out_dir=gin.REQUIRED,
                 is_preprocess=gin.REQUIRED,
                 max_image_large_side=1280,
                 max_text_size=800,
                 min_text_size=5,
                 min_crop_side_ratio=0.1,
                 geometry="RBOX",
                 number_images_per_tfrecords=8,
                 num_cores=4,
                 batch_size=4,
                 prefetch_size=16,
                 num_epochs=5):
        """
        """

        IDataset.__init__(self,
                          data_in_dir=data_in_dir,
                          data_out_dir=data_out_dir,
                          batch_size=batch_size,
                          is_preprocess=is_preprocess,
                          num_cores=4)

        self._data_in_dir = data_in_dir

        self._train_out_dir = data_out_dir + "/train/"
        self._val_out_dir = data_out_dir + "/val/"
        self._test_out_dir = data_out_dir + "/test/"

        make_dirs(self._train_out_dir)
        make_dirs(self._val_out_dir)
        make_dirs(self._test_out_dir)

        self._geometry = geometry
        self._min_text_size = min_text_size
        self._max_image_large_side = max_image_large_side
        self._max_text_size = max_text_size
        self._min_crop_side_ratio = min_crop_side_ratio
        self._number_images_per_tfrecords = number_images_per_tfrecords

        self._data_in_dir = data_in_dir
        self._num_cores = num_cores
        self._batch_size = batch_size
        self._prefetch_size = prefetch_size

        self._num_train_examples = 0

        self._num_epochs = num_epochs

        if self._is_preprocess is True:
            self.preprocess()

    def get_tf_records_count(self, dir):
        if not os.path.exists(dir):
            raise FileNotFoundError("Enable preprocess to get sample counts from TFRecords!")

        # TODO find a right way to get this
        path = os.path.join(dir, "*.tfrecords")
        path = path.replace("//", "/")
        files = glob.glob(pathname=path)
        return get_tf_records_count(files=files)

    def _get_num_train_samples(self):
        # return self.get_tf_records_count(self._train_out_dir)
        images = get_images(self._data_in_dir + "/train/")
        return len(images)

    def _get_num_val_samples(self):
        # return self.get_tf_records_count(self._val_out_dir)
        images = get_images(self._data_in_dir + "/val/")
        return len(images)

    def _get_num_test_samples(self):
        # return self.get_tf_records_count(self._test_out_dir)
        images = get_images(self._data_in_dir + "/test/")
        return len(images)

    def _get_features(self, image_mat, score_map_mat, geo_map_mat):#, training_masks_mat):
        """
        Given different features matrices, this routine wraps the matrices as TF features
        """
        return {
            "images": _mat_feature(image_mat),
            "score_maps": _mat_feature(score_map_mat),
            "geo_maps": _mat_feature(geo_map_mat),
            # "training_masks": _mat_feature(training_masks_mat)
        }

    def write_tf_records(self, images, file_path_name):
        """
        Uses sub routine to create TF records files from list of images and corresponding
        text files with text polygon regions (inffered from image file names)
        :param images: List of image files
        :param file_path_name: TF record file path
        :return:
        """
        num_of_files_skipped = 0

        with tf.io.TFRecordWriter(file_path_name) as writer:
            for image_file in tqdm(images, desc="pid : " + str(os.getpid())):
                ret = image_2_data(image_file_path=image_file,
                                   geometry=self._geometry,
                                   min_text_size=self._min_text_size,
                                   min_crop_side_ratio=self._min_crop_side_ratio)
                try:
                    # image_mat, score_map_mat, geo_map_mat, training_masks_mat = ret
                    image_mat, score_map_mat, geo_map_mat = ret
                except:
                    num_of_files_skipped += 1
                    print("Number of files skipped : ", num_of_files_skipped)
                    continue
                features = tf.train.Features(
                    feature=self._get_features(image_mat, score_map_mat, geo_map_mat))#, training_masks_mat))
                example = tf.train.Example(features=features)
                writer.write(example.SerializeToString())

    def task(self, images_out_path_tuple):
        self.write_tf_records(images=images_out_path_tuple[0], file_path_name=images_out_path_tuple[1])

    def prepare_data(self, data_path, out_path):

        print("Serializing data found in ", data_path)

        images = get_images(data_path)

        index = 0
        multiprocess_list = [] # list of tuples: list of images and a TFRecord file name

        for i in range(0, len(images), self._number_images_per_tfrecords):
            file_path_name = out_path + "/" + str(index) + ".tfrecords"
            if os.path.exists(file_path_name):
                # num_records = get_tf_records_count([file_path_name])
                print_debug("Found in " + str(file_path_name) + f"with records already! Hence skipping")
            else:
                multiprocess_list.append((images[i:i + self._number_images_per_tfrecords], file_path_name))
                index += 1

        if len(multiprocess_list) > 1: #process only when no TFRecords are found
            # creating a pool object
            pool = multiprocessing.Pool()

            # map list to target function
            pool.map(self.task, multiprocess_list)

            pool.close()
            pool.join()

    def preprocess(self):
        self.prepare_data(data_path=self._data_in_dir + "/train/", out_path=self._train_out_dir)
        self.prepare_data(data_path=self._data_in_dir + "/val/", out_path=self._val_out_dir)
        self.prepare_data(data_path=self._data_in_dir + "/test/", out_path=self._test_out_dir)

    def decode(self, serialized_example):
        # 1. define a parser
        features = tf.io.parse_single_example(
            serialized_example,
            # Defaults are not specified since both keys are required.
            features={
                'images': tf.io.FixedLenFeature([512 * 512 * 3], tf.float32),
                'score_maps': tf.io.FixedLenFeature([128 * 128 * 1], tf.float32),
                'geo_maps': tf.io.FixedLenFeature([128 * 128 * 5], tf.float32),
                # 'training_masks': tf.io.FixedLenFeature([128 * 128 * 1], tf.float32),
            })

        image = tf.reshape(
            tf.cast(features['images'], tf.float32), shape=[512, 512, 3])
        score_map = tf.reshape(
            tf.cast(features['score_maps'], tf.float32), shape=[128, 128, 1])
        geo_map = tf.reshape(
            tf.cast(features['geo_maps'], tf.float32), shape=[128, 128, 5])

        # training_masks = tf.reshape(
        #     tf.cast(features['training_masks'], tf.float32), shape=[128, 128, 1])
        #
        # return {"images": image, "score_maps": score_map, "geo_maps": geo_map,
        #         "training_masks": training_masks}, training_masks #dummy label/Y

        return {"images": image, "score_maps": score_map, "geo_maps": geo_map}, image #dummy label/Y

    def get_tf_train_dataset(self):
        # print_info("_get_train_dataset {}".format(num_epochs))
        memory_usage_psutil()
        path = os.path.join(self._train_out_dir, "*.tfrecords")
        path = path.replace("//", "/")
        # files = glob.glob(pathname=path)

        train_tfrecord_files = tf.data.Dataset.list_files(path)

        # TF dataset APIs
        # dataset = tf.data.TFRecordDataset(files, num_parallel_reads=self._num_cores)
        dataset = train_tfrecord_files.interleave(
            tf.data.TFRecordDataset, cycle_length=self._num_cores,
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.shuffle(self._batch_size * 10, 42)
        # Map the generator output as features as a dict and labels
        dataset = dataset.map(map_func=self.decode, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(batch_size=self._batch_size, drop_remainder=False)
        dataset= dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        # self._train_dataset = self._train_dataset.make_one_shot_iterator()
        dataset = dataset.repeat(self._num_epochs)
        # dataset = dataset.cache(filename=os.path.join(self.iterator_dir, "train_data_cache"))
        print_info("Dataset output sizes are: ")
        print_info(dataset)
        memory_usage_psutil()

        # return dataset

        # iterator = self._train_dataset.make_one_shot_iterator()
        #
        # batch_feats, batch_labels = iterator.get_next()

        return dataset

    def get_tf_val_dataset(self):
        memory_usage_psutil()
        path = os.path.join(self._val_out_dir, "*.tfrecords")
        path = path.replace("//", "/")
        # train_tfrecord_files = glob.glob(pathname=path)

        val_tfrecord_files = tf.data.Dataset.list_files(path)

        # TF dataset APIs
        # dataset = tf.data.TFRecordDataset(files, num_parallel_reads=self._num_cores)
        dataset = val_tfrecord_files.interleave(
            tf.data.TFRecordDataset, cycle_length=self._num_cores,
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.shuffle(self._batch_size * 10, 42)
        # Map the generator output as features as a dict and labels
        dataset = dataset.map(map_func=self.decode, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(batch_size=self._batch_size, drop_remainder=False)
        self._val_dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        # self._val_dataset = self._val_dataset.make_one_shot_iterator()

        dataset = dataset.repeat()
        # dataset = dataset.cache(filename=os.path.join(self.iterator_dir, "train_data_cache"))

        print_info("Dataset output sizes are: ")
        print_info(self._val_dataset)
        memory_usage_psutil()
        # iterator = self._val_dataset.make_one_shot_iterator()
        #
        # batch_feats, batch_labels = iterator.get_next()

        return dataset

    def get_tf_test_dataset(self):
        """
        Reads TFRecords, decode and batches them
        :return: callable
        """

        path = os.path.join(self._test_out_dir, "*.tfrecords")
        path = path.replace("//", "/")
        files = glob.glob(pathname=path)

        assert len(files) > 0

        # TF dataset APIs
        # dataset = tf.data.TFRecordDataset(files, num_parallel_reads=self._num_cores)

        files = tf.data.Dataset.list_files(path)

        # TF dataset APIs
        # dataset = tf.data.TFRecordDataset(files, num_parallel_reads=self._num_cores)
        dataset = files.interleave(
            tf.data.TFRecordDataset, cycle_length=self._num_cores,
            num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # Map the generator output as features as a dict and labels
        dataset = dataset.map(self.decode)

        dataset = dataset.batch(
            batch_size=self._hparams.batch_size, drop_remainder=False)
        # dataset = dataset.shuffle(self._prefetch_size * 2, 42)
        dataset = dataset.prefetch(self._prefetch_size)
        # dataset = dataset.repeat()
        print_info("Dataset output sizes are: ")
        print_info(dataset)
        return dataset

    def get_tf_serving_dataset(self):
        inputs = {
            # "images": tf.Variable(dtype=tf.float32, shape=[None, None, None, 3], validate_shape=False),
            "images": tf.compat.v1.placeholder(tf.float32, [None, None, None, 3]),
        }
        return tf.estimator.export.ServingInputReceiver(inputs, inputs)

    def get_train_dataset_gen(self, num_epochs=None):
        """
        Returns
        an data set generator function that can be used in train loop
        :return:
        """
        return generator(data_dir=self._data_in_dir + "/train",
                         batch_size=self.batch_size,
                         geometry="RBOX",
                         min_text_size=self._min_text_size,
                         min_crop_side_ratio=self._min_crop_side_ratio,
                         input_size=512,
                         background_ratio=3. / 8,
                         random_scale=np.array([0.5, 1, 2.0]),  # , 3.0]),
                         vis=False,
                         is_train=True,
                         shuffle=True)


    def get_val_dataset_gen(self, num_epochs=None):
        """
        Returns an data set generator function that can be used in validation loop
        :return:
        """
        return generator(data_dir=self._data_in_dir + "/val",
                         batch_size=self.batch_size,
                         geometry="RBOX",
                         min_text_size=self._min_text_size,
                         min_crop_side_ratio=self._min_crop_side_ratio,
                         input_size=512,
                         background_ratio=3. / 8,
                         random_scale=np.array([0.5, 1, 2.0]),  # , 3.0]),
                         vis=False,
                         is_train=True,
                         shuffle=True)

    def get_test_dataset_gen(self):
        """
        Returns an data set generator function that can be used in test loop
        :return:
        """
        return generator(data_dir=self._data_in_dir + "/test",
                         batch_size=self.batch_size,
                         geometry="RBOX",
                         min_text_size=self._min_text_size,
                         min_crop_side_ratio=self._min_crop_side_ratio,
                         input_size=512,
                         background_ratio=3. / 8,
                         random_scale=np.array([0.5, 1, 2.0]),  # , 3.0]),
                         vis=False,
                         is_train=False,
                         shuffle=True)